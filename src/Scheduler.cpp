#include "Scheduler.h"
#include "utils/include/logger.h"

void Scheduler::schedule() {
    
    while(True){

        launchSequence();

        if(!prefilling_queue.empty() || !decoding_queue.empty()){
            Batch decode_batch = buildDecodeBatch();
            
            model.decode_forward(decode_batch);

        }
        

        if(!waiting_queue.empty()){
            Batch prefill_batch = buildPrefillBatch();
            model.prefill_forward(prefill_batch);

            movePrefilledToDecoding(prefill_batch);
        }

        handleFinishedSequence();
        returnSequenceOutput();
    }
}

ErrorCode Scheduler::movePrefilledToDecoding(const Batch& prefill_batch) {
    for(auto it = prefill_batch.sequences.begin(); it != prefill_batch.sequences.end();){
        if((*it)->SequenceState == SequenceState::PREFILLED){
            (*it)->SequenceState = SequenceState::DECODING;
            LOG_DEBUG("Sequence " + std::to_string((*it)->seq_id) + " moved to DECODING state.");
            decoding_queue.push_back(*it);
            it = prefilling_queue.erase(it);
        } else {
            ++it;
        }
    }
    return ErrorCode::SUCCESS;
}

variant<Batch, ErrorCode> Scheduler::buildDecodeBatch() {
    // Implement the logic to build a batch of sequences for processing
    Batch batch;
    batch.batch_size = 0;
    
    for(auto& seq : decoding_queue){
        if(seq->SequenceState!= SequenceState::FINISHED){
            batch.token_ids.insert(batch.token_ids.end(), seq->token_ids.begin(), seq->token_ids.end());

            size_t seq_len = seq->token_ids.size();
            for(size_t i = 0; i < seq_len; ++i){
                batch.token_positions.push_back(i);
                batch.sequences.push_back(seq);
            }
            batch.num_tokens += seq->token_ids.size();
            batch.batch_size++;

            if(seq->sequence_len % BLOCK_SIZE == 0){
                variant<CacheBlock*, ErrorCode> result = cache_manager->allocate_cache_block();
                if(std::holds_alternative<CacheBlock*>(result)){
                    seq->blocks.push_back(std::get<CacheBlock*>(result));
                } else {
                    return ErrorCode::MEMORY_FAILURE;
                }
            }
            
            if(batch.batch_size >= MAX_DECODE_BATCH_SIZE){
                break;
            }
        }
    }
    while(batch.batch_size < MAX_DECODE_BATCH_SIZE && !prefilling_queue.empty()){
        Sequence seq = prefilling_queue.front();
        if(seq.SequenceState == SequenceState::PREFILLED){
            prefilling_queue.erase(prefilling_queue.begin());
            seq.SequenceState = SequenceState::DECODING;
            decoding_queue.push_back(seq);
            LOG_DEBUG("Sequence " + std::to_string(seq.seq_id) + " moved from PREFILLED to DECODING state.");

            batch.token_ids.insert(batch.token_ids.end(), seq.token_ids.begin(), seq.token_ids.end());

            size_t seq_len = seq.token_ids.size();
            for(size_t i = 0; i < seq_len; ++i){
                batch.token_positions.push_back(i);
                batch.sequences.push_back(&seq);
            }
            batch.num_tokens += seq.token_ids.size();
            batch.batch_size++;

            if(seq.sequence_len % BLOCK_SIZE == 0){
                variant<CacheBlock*, ErrorCode> result = cache_manager->allocate_cache_block();
                if(std::holds_alternative<CacheBlock*>(result)){
                    seq.blocks.push_back(std::get<CacheBlock*>(result));
                } else {
                    return ErrorCode::MEMORY_FAILURE;
                }
            }
        }
    }
    return std::make_pair(batch, ErrorCode::SUCCESS);
}

variant<Batch, ErrorCode> Scheduler::buildPrefillBatch() {
    Batch batch;
    batch.batch_size = 0;
    
    for(auto& seq : waiting_queue){
        if(seq->SequenceState == SequenceState::WAITING){
            waiting_queue.erase(waiting_queue.begin());
            seq->SequenceState = SequenceState::PREFILLING;
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
            
            //allocate cache blocks for the sequence prefill
            size_t num_blocks = (seq->seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
            for(size_t i = 0; i < num_blocks; ++i){
                variant<CacheBlock*, ErrorCode> result = cache_manager->allocate_cache_block();
                if(std::holds_alternative<CacheBlock*>(result)){
                    seq->blocks.push_back(std::get<CacheBlock*>(result));
                } else {
                    // Handle cache allocation failure (e.g., log an error, skip the sequence, etc.)
                    return ErrorCode::MEMORY_FAILURE;
                }
            }

            if(batch.batch_size >= MAX_PREFILL_BATCH_SIZE){
                break;
            }
        }
    }
    return std::make_pair(batch, ErrorCode::SUCCESS);
}

ErrorCode Scheduler::addSequence(size_t seq_id, vector<size_t> token_ids) {
    auto new_seq = std::make_shared<Sequence>(seq_id);
    new_seq->token_ids = token_ids;
    waiting_queue.push_back(new_seq);
    LOG_DEBUG("Sequence added to waiting queue: " + std::to_string(new_seq->seq_id));
    return ErrorCode::SUCCESS;
}


ErrorCode Scheduler::launchSequence(){
    while(!waiting_queue.empty()){
        auto seq = waiting_queue.front();
        waiting_queue.erase(waiting_queue.begin());
        seq->SequenceState = SequenceState::PREFILLING;
        prefilling_queue.push_back(seq);
        LOG_DEBUG("Sequence " + std::to_string(seq->seq_id) + " moved to PREFILLING state.");
    }
    return ErrorCode::SUCCESS;
}

ErrorCode Scheduler::handleFinishedSequence(){
    for(auto it = decoding_queue.begin(); it != decoding_queue.end();){
        if((*it)->SequenceState == SequenceState::FINISHED){
            // Handle the finished sequence (e.g., remove from decoding queue, update cache, etc.)
            finished_queue.push_back(*it);
            LOG_DEBUG("Sequence " + std::to_string((*it)->seq_id) + " moved to FINISHED state.");
            it = decoding_queue.erase(it);
        } else {
            ++it;
        }
    }
    return ErrorCode::SUCCESS;
}

ErrorCode Scheduler::returnSequenceOutput() {
    for(auto& seq : finished_queue){
        if(seq->SequenceState == SequenceState::FINISHED){
            std::lock_guard<std::mutex> lock(seq->mtx);
            seq->cv.notify_one(); 

            //delete cache blocks associated with the sequence
            for(auto& block : seq->blocks){
                cache_manager->free_cache_block(block->block_id);
            }


        }
    }
    return ErrorCode::SUCCESS;
}

ErrorCode Scheduler::getSequenceById(size_t seq_id, Sequence* seq) {
    seq = nullptr;
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

ErrorCode Scheduler::getFinishedSequenceById(size_t seq_id, Sequence* seq) {
    seq = nullptr;
    for (auto& sequence : finished_queue) {
        if (sequence->seq_id == seq_id) {
            seq = sequence;
            return ErrorCode::SUCCESS;
        }
    }
    return ErrorCode::SEQUENCE_NOT_FOUND;
}

ErrorCode Scheduler::removeFinishedSequenceById(size_t seq_id) {
    for (auto it = finished_queue.begin(); it != finished_queue.end(); ++it) {
        if (it->seq_id == seq_id) {
            finished_queue.erase(it);
            LOG_DEBUG("Sequence " + std::to_string(seq_id) + " removed from finished queue.");
            return ErrorCode::SUCCESS;
        }
    }
    return ErrorCode::SEQUENCE_NOT_FOUND;
}