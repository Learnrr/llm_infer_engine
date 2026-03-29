#include "Scheduler.h"
#include "utils/logger.h"

bool Scheduler::hasPendingWorkLocked() const {
    return !prepared_queue.empty() ||
           !waiting_queue.empty() ||
           !prefilling_queue.empty() ||
           !decoding_queue.empty();
}

void Scheduler::schedule() {
    while(!stop_requested.load()){
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            queue_cv.wait(lock, [this]() {
                //when stop or has pending work
                return stop_requested.load() || hasPendingWorkLocked();
            });
            if (stop_requested.load()) {
                break;
            }
        }

        launchSequence();

        bool has_decode_work = false;
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            has_decode_work = !prefilling_queue.empty() || !decoding_queue.empty();
        }
        if(has_decode_work){
            auto result = buildDecodeBatch();
            if (std::holds_alternative<ErrorCode>(result)) {
                LOG_ERROR("Failed to build decode batch.");
                return;
            } 
            Batch decode_batch = std::get<Batch>(result);
            if (model == nullptr || workspace == nullptr) {
                LOG_ERROR("Scheduler decode path has null model/workspace");
                return;
            }
            LOG_DEBUG(
                "Calling model->decode_forward: num_tokens=" + std::to_string(decode_batch.num_tokens) +
                ", token_ids=" + std::to_string(decode_batch.token_ids.size()) +
                ", token_positions=" + std::to_string(decode_batch.token_positions.size()) +
                ", sequences=" + std::to_string(decode_batch.sequences.size())
            );
            model->decode_forward(decode_batch, *workspace);
            appendDecodedTokens(decode_batch);

            moveDecodingToFinished(decode_batch);
        }

        
        

        bool has_waiting_work = false;
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            has_waiting_work = !waiting_queue.empty();
        }
        if(has_waiting_work){
            auto result = buildPrefillBatch();
            if (std::holds_alternative<ErrorCode>(result)) {
                LOG_ERROR("Failed to build prefill batch.");
                return;
            }
            Batch prefill_batch = std::get<Batch>(result);
            if (model == nullptr || workspace == nullptr) {
                LOG_ERROR("Scheduler prefill path has null model/workspace");
                return;
            }
            LOG_DEBUG(
                "Calling model->prefill_forward: num_tokens=" + std::to_string(prefill_batch.num_tokens) +
                ", token_ids=" + std::to_string(prefill_batch.token_ids.size()) +
                ", token_positions=" + std::to_string(prefill_batch.token_positions.size()) +
                ", sequences=" + std::to_string(prefill_batch.sequences.size())
            );
            model->prefill_forward(prefill_batch, *workspace);
            LOG_DEBUG("Returned from model->prefill_forward");

            for (const auto& seq : prefill_batch.sequences) {
                if (seq && seq->state == SequenceState::PREFILLING) {
                    seq->state = SequenceState::PREFILLED;
                }
            }

            movePrefilledToDecoding(prefill_batch);
        }

        handleFinishedSequence();
        returnSequenceOutput();
    }
}

void Scheduler::request_stop() {
    stop_requested.store(true);
    queue_cv.notify_all();
}

ErrorCode Scheduler::moveDecodingToFinished(const Batch& decode_batch) {
    for(auto& seq : decode_batch.sequences){
        if(seq->state == SequenceState::DECODING){
            if(seq->token_ids.size() >= engine_config.max_sequence_length 
            || seq->token_ids.back() == engine_config.model_config.eos_token_id){
                seq->state = SequenceState::FINISHED;
                LOG_DEBUG("Sequence " + std::to_string(seq->seq_id) + " moved to FINISHED state.");
            }
        }
    }
    return ErrorCode::SUCCESS;
}

void Scheduler::appendDecodedTokens(Batch& decode_batch) {
    if (decode_batch.sampled_token_ids.size() != decode_batch.sequences.size()) {
        // Model has not attached one next-token per sequence yet.
        return;
    }

    for (size_t i = 0; i < decode_batch.sequences.size(); ++i) {
        auto& seq = decode_batch.sequences[i];
        if (!seq) {
            continue;
        }
        seq->add_token(decode_batch.sampled_token_ids[i]);
    }
}

ErrorCode Scheduler::movePrefilledToDecoding(const Batch& prefill_batch) {
    std::lock_guard<std::mutex> lock(queue_mutex);
    for (const auto& seq : prefill_batch.sequences) {
        if (!seq) {
            continue;
        }
        if (seq->state == SequenceState::PREFILLED) {
            seq->state = SequenceState::DECODING;
            LOG_DEBUG("Sequence " + std::to_string(seq->seq_id) + " moved to DECODING state.");
            decoding_queue.push_back(seq);

            for (auto qit = prefilling_queue.begin(); qit != prefilling_queue.end(); ++qit) {
                if (*qit == seq) {
                    prefilling_queue.erase(qit);
                    break;
                }
            }
        }
    }
    return ErrorCode::SUCCESS;
}

std::variant<Batch, ErrorCode> Scheduler::buildDecodeBatch() {
    // Implement the logic to build a batch of sequences for processing
    std::lock_guard<std::mutex> lock(queue_mutex);
    Batch batch;
    batch.batch_size = 0;
    batch.num_tokens = 0;
    
    for(auto& seq : decoding_queue){
        if(seq->state!= SequenceState::FINISHED){
            if(seq->token_ids.empty()){
                continue;
            }

            size_t last_pos = seq->token_ids.size() - 1;
            //decode batch does not have all previous token_ids, 
            //read-cache reads all previous tokens KV up to current position.
            batch.token_ids.push_back(seq->token_ids[last_pos]);
            batch.token_positions.push_back(last_pos);
            batch.sequences.push_back(seq);
            batch.num_tokens += 1;  
            batch.batch_size++;

            if(seq->seq_len % engine_config.block_size == 0){
                std::variant<std::shared_ptr<CacheBlock>, ErrorCode> result = cache_manager->allocate_cache_block();
                if(std::holds_alternative<std::shared_ptr<CacheBlock>>(result)){
                    seq->blocks.push_back(std::get<std::shared_ptr<CacheBlock>>(result));
                } else {
                    return ErrorCode::MEMORY_FAILURE;
                }
            }
            
            if(batch.batch_size >= engine_config.max_decode_batch_size){
                break;
            }
        }
    }
    while(batch.batch_size < engine_config.max_decode_batch_size && !prefilling_queue.empty()){
        auto seq = prefilling_queue.front();
        if(seq->state == SequenceState::PREFILLED){
            prefilling_queue.erase(prefilling_queue.begin());
            seq->state = SequenceState::DECODING;
            decoding_queue.push_back(seq);
            LOG_DEBUG("Sequence " + std::to_string(seq->seq_id) + " moved from PREFILLED to DECODING state.");

            if(seq->token_ids.empty()){
                continue;
            }

            size_t last_pos = seq->token_ids.size() - 1;
            batch.token_ids.push_back(seq->token_ids[last_pos]);
            batch.token_positions.push_back(last_pos);
            batch.sequences.push_back(seq);
            batch.num_tokens += 1;
            batch.batch_size++;

            if(seq->seq_len % engine_config.block_size == 0){
                std::variant<std::shared_ptr<CacheBlock>, ErrorCode> result = cache_manager->allocate_cache_block();
                if(std::holds_alternative<std::shared_ptr<CacheBlock>>(result)){
                    seq->blocks.push_back(std::get<std::shared_ptr<CacheBlock>>(result));
                } else {
                    return ErrorCode::MEMORY_FAILURE;
                }
            }
        }
    }
    return batch;
}

std::variant<Batch, ErrorCode> Scheduler::buildPrefillBatch() {
    std::lock_guard<std::mutex> lock(queue_mutex);
    Batch batch;
    batch.batch_size = 0;
    batch.num_tokens = 0;
    
    for (auto it = waiting_queue.begin(); it != waiting_queue.end();) {
        auto seq = *it;
        if(seq->state == SequenceState::WAITING){
            it = waiting_queue.erase(it);
            seq->state = SequenceState::PREFILLING;
            prefilling_queue.push_back(seq);
            LOG_DEBUG("Sequence " + std::to_string(seq->seq_id) + " moved to PREFILLING state.");

            batch.token_ids.insert(batch.token_ids.end(), seq->token_ids.begin(), seq->token_ids.end());
            size_t seq_len = seq->token_ids.size();
            for(size_t i = 0; i < seq_len; ++i){
                batch.token_positions.push_back(i);
                batch.sequences.push_back(seq);
            }
            batch.num_tokens += seq->token_ids.size();
            batch.batch_size++;

            LOG_DEBUG("Sequence " + std::to_string(seq->seq_id) + " added to prefill batch with " + std::to_string(seq_len) + " tokens.");
            
            //allocate cache blocks for the sequence prefill
            size_t num_blocks = (seq->seq_len + engine_config.block_size - 1) / engine_config.block_size;
            for(size_t i = 0; i < num_blocks; ++i){
                auto result = cache_manager->allocate_cache_block();
                if(std::holds_alternative<std::shared_ptr<CacheBlock>>(result)){
                    seq->blocks.push_back(std::get<std::shared_ptr<CacheBlock>>(result));
                } else {
                    // Handle cache allocation failure (e.g., log an error, skip the sequence, etc.)
                    return ErrorCode::MEMORY_FAILURE;
                }
            }
            LOG_DEBUG("Sequence " + std::to_string(seq->seq_id) + " allocated " + std::to_string(num_blocks) + " cache blocks for prefill.");
            if(batch.batch_size >= engine_config.max_prefill_batch_size){
                break;
            }

            continue;
        }
        ++it;
    }
    return batch;
}

//this will create a new sequence in the scheduler and add it to queue
ErrorCode Scheduler::addSequence(size_t seq_id, std::vector<size_t> token_ids) {
    auto new_seq = std::make_shared<Sequence>(seq_id);
    new_seq->token_ids = token_ids;
    new_seq->seq_len = token_ids.size();
    new_seq->state = SequenceState::PREPARED;
    new_seq->blocks.clear();
    std::lock_guard<std::mutex> lock(queue_mutex);
    prepared_queue.push_back(new_seq);
    LOG_DEBUG("Sequence added to prepared queue: " + std::to_string(new_seq->seq_id));
    queue_cv.notify_one();
    return ErrorCode::SUCCESS;
}


ErrorCode Scheduler::launchSequence(){
    std::lock_guard<std::mutex> lock(queue_mutex);
    while(!prepared_queue.empty()){
        auto seq = prepared_queue.front();
        prepared_queue.erase(prepared_queue.begin());
        seq->state = SequenceState::WAITING;
        waiting_queue.push_back(seq);
        LOG_DEBUG("Sequence " + std::to_string(seq->seq_id) + " moved to WAITING state.");
    }
    return ErrorCode::SUCCESS;
}

ErrorCode Scheduler::handleFinishedSequence(){
    std::lock_guard<std::mutex> lock(queue_mutex);
    for(auto it = decoding_queue.begin(); it != decoding_queue.end();){
        if((*it)->state == SequenceState::FINISHED){
            // Handle the finished sequence (e.g., remove from decoding queue, update cache, etc.)
            finished_queue.push_back(*it);
            LOG_DEBUG("Sequence " + std::to_string((*it)->seq_id) + " moved to FINISHED queue.");
            it = decoding_queue.erase(it);
        } else {
            ++it;
        }
    }
    return ErrorCode::SUCCESS;
}

ErrorCode Scheduler::returnSequenceOutput() {
    std::lock_guard<std::mutex> queue_lock(queue_mutex);
    for(auto& seq : finished_queue){
        if(seq->state == SequenceState::FINISHED && !seq->finish_handled){
            std::lock_guard<std::mutex> lock(seq->mtx);
            seq->cv.notify_one(); 

            //delete cache blocks associated with the sequence
            for(auto& block : seq->blocks){
                cache_manager->free_cache_block(block->block_id);
            }

            seq->finish_handled = true;
            
        }
    }
    return ErrorCode::SUCCESS;
}

ErrorCode Scheduler::getSequenceById(size_t seq_id, std::shared_ptr<Sequence>& seq) {
    seq = nullptr;
    std::lock_guard<std::mutex> lock(queue_mutex);
    for(auto& sequence : prepared_queue) {
        if (sequence->seq_id == seq_id) {
            seq = sequence;
            return ErrorCode::SUCCESS;
        }
    }
    for (auto& sequence : waiting_queue) {
        if (sequence->seq_id == seq_id) {
            seq = sequence;
            return ErrorCode::SUCCESS;
        }
    }
    for (auto& sequence : prefilling_queue) {
        if (sequence->seq_id == seq_id) {
            seq = sequence;
            return ErrorCode::SUCCESS;
        }
    }
    for (auto& sequence : decoding_queue) {
        if (sequence->seq_id == seq_id) {
            seq = sequence;
            return ErrorCode::SUCCESS;
        }
    }
    for (auto& sequence : finished_queue) {
        if (sequence->seq_id == seq_id) {
            seq = sequence;
            return ErrorCode::SUCCESS;
        }
    }
    return ErrorCode::SEQUENCE_NOT_FOUND;
}

ErrorCode Scheduler::getFinishedSequenceById(size_t seq_id, std::shared_ptr<Sequence>& seq) {
    seq = nullptr;
    std::lock_guard<std::mutex> lock(queue_mutex);
    for (auto& sequence : finished_queue) {
        if (sequence->seq_id == seq_id) {
            seq = sequence;
            return ErrorCode::SUCCESS;
        }
    }
    return ErrorCode::SEQUENCE_NOT_FOUND;
}

ErrorCode Scheduler::removeFinishedSequenceById(size_t seq_id) {
    std::lock_guard<std::mutex> lock(queue_mutex);
    for (auto it = finished_queue.begin(); it != finished_queue.end(); ++it) {
        if (*it != nullptr && (*it)->seq_id == seq_id) {
            finished_queue.erase(it);
            LOG_DEBUG("Sequence " + std::to_string(seq_id) + " removed from finished queue.");
            return ErrorCode::SUCCESS;
        }
    }
    return ErrorCode::SEQUENCE_NOT_FOUND;
}