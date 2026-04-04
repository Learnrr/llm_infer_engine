#include "Scheduler.h"
#include "executor/CoordinatorExecutor.h"
#include "channel/ChannelManager.h"
#include "utils/logger.h"
#include "utils/timer.h"

#include <unordered_set>

void Scheduler::set_channels() {
    ChannelManager* manager = ChannelManager::get_instance();
    auto get_or_null = [manager](const std::string& name) -> Channel* {
        Channel* channel = nullptr;
        ErrorCode err = manager->get_channel(name, channel);
        if (err != ErrorCode::SUCCESS) {
            return nullptr;
        }
        return channel;
    };

    const int last_rank = engine_config.world_size - 1;
    Channel* to_worker0 = get_or_null("scheduler_to_worker_0");
    Channel* from_worker0 = get_or_null("worker_0_to_scheduler");
    Channel* to_worker_last = get_or_null("scheduler_to_worker_" + std::to_string(last_rank));
    Channel* from_worker_last = get_or_null("worker_" + std::to_string(last_rank) + "_to_scheduler");

    CoordinatorExecutor* coordinator = dynamic_cast<CoordinatorExecutor*>(model_executor.get());
    if (coordinator != nullptr) {
        coordinator->set_channels(from_worker0, to_worker0, from_worker_last, to_worker_last);
    }
}

void Scheduler::run() {
    LOG_INFO("Scheduler started running.");
    schedule();
    LOG_INFO("Scheduler stopped running.");
}

bool Scheduler::hasPendingWorkLocked() const {
    return !prepared_queue.empty() ||
           !waiting_queue.empty() ||
           !prefilling_queue.empty() ||
           !decoding_queue.empty();
}

void Scheduler::schedule() {
    while(!stop_requested.load()) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            queue_cv.wait(lock, [this]() {
                return stop_requested.load() || hasPendingWorkLocked();
            });
            if (stop_requested.load()) {
                stopWorkers();
                break;
            }
        }

        launchSequence();

        bool has_decode_work = false;
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            has_decode_work = !prefilling_queue.empty() || !decoding_queue.empty();
        }
        if (has_decode_work) {
            auto result = buildDecodeBatch();
            if (std::holds_alternative<ErrorCode>(result)) {
                LOG_ERROR("Failed to build decode batch.");
                return;
            }
            Batch decode_batch = std::get<Batch>(result);
            if(decode_batch.batch_size == 0 || decode_batch.num_tokens == 0){
                continue;
            }
            if (model_executor == nullptr) {
                LOG_ERROR("Scheduler decode path has null model executor");
                return;
            }

            // model executor in scheduler will coordinate with workers to run the decode batch
            model_executor->run_decode(decode_batch);
            bool decode_ok = true;
            if (CoordinatorExecutor* coordinator = dynamic_cast<CoordinatorExecutor*>(model_executor.get())) {
                decode_ok = coordinator->consume_last_forward_ok();
            }
            if (!decode_ok) {
                LOG_ERROR("Scheduler decode forward failed; skipping state transition for this batch.");
                if (CoordinatorExecutor* coordinator = dynamic_cast<CoordinatorExecutor*>(model_executor.get())) {
                    coordinator->run_release_events(decode_batch);
                }
                continue;
            }

            if (CoordinatorExecutor* coordinator = dynamic_cast<CoordinatorExecutor*>(model_executor.get())) {
                coordinator->run_release_events(decode_batch);
            }
            appendDecodedTokens(decode_batch);

            moveDecodingToFinished(decode_batch);
        }

        bool has_waiting_work = false;
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            has_waiting_work = !waiting_queue.empty();
        }
        if (has_waiting_work) {
            auto result = buildPrefillBatch();
            if (std::holds_alternative<ErrorCode>(result)) {
                LOG_ERROR("Failed to build prefill batch.");
                return;
            }
            Batch prefill_batch = std::get<Batch>(result);
            if(prefill_batch.batch_size == 0 || prefill_batch.num_tokens == 0){
                continue;
            }
            if (model_executor == nullptr) {
                LOG_ERROR("Scheduler prefill path has null model executor");
                return;
            }
            // model executor in scheduler will coordinate with workers to run the prefill batch
            model_executor->run_prefill(prefill_batch);
            bool prefill_ok = true;
            if (CoordinatorExecutor* coordinator = dynamic_cast<CoordinatorExecutor*>(model_executor.get())) {
                prefill_ok = coordinator->consume_last_forward_ok();
            }
            if (!prefill_ok) {
                LOG_ERROR("Scheduler prefill forward failed; recovering affected sequences to WAITING.");
                recoverFromPrefillFailure(prefill_batch);
                continue;
            }

            if (CoordinatorExecutor* coordinator = dynamic_cast<CoordinatorExecutor*>(model_executor.get())) {
                coordinator->run_release_events(prefill_batch);
            }
            LOG_DEBUG("Returned from model_executor->run_prefill");

            for (size_t seq_id : prefill_batch.sequence_ids) {
                auto seq = seq_pool->get(seq_id);
                if (seq && seq->state == SequenceState::PREFILLING) {
                    seq->state = SequenceState::PREFILLED;
                }
            }

            movePrefilledToDecoding(prefill_batch);
        }
        //check if there are finished sequences in decoding queue
        handleFinishedSequence();

        //return sequence output to upper layer,
        // and notify workers to free cache for finished sequences
        returnSequenceOutput();
    }
}

void Scheduler::request_stop() {
    stop_requested.store(true);
    queue_cv.notify_all();
}

void Scheduler::stopWorkers() {
    CoordinatorExecutor* coordinator = dynamic_cast<CoordinatorExecutor*>(model_executor.get());
    if (coordinator == nullptr) {
        LOG_ERROR("Scheduler stop requested but coordinator executor is null");
        return;
    }
    coordinator->run_stop();
}

void Scheduler::recoverFromPrefillFailure(const Batch& prefill_batch) {
    std::lock_guard<std::mutex> lock(queue_mutex);
    std::unordered_set<size_t> recovered;
    for (size_t seq_id : prefill_batch.sequence_ids) {
        if (!recovered.insert(seq_id).second) {
            continue;
        }

        auto seq = seq_pool->get(seq_id);
        if (seq == nullptr) {
            continue;
        }
        //failed prefill should be moved back to waiting queue for next entry
        if (seq->state != SequenceState::PREFILLING) {
            continue;
        }
        seq->state = SequenceState::WAITING;
        waiting_queue.push_back(seq_id);

        for (auto it = prefilling_queue.begin(); it != prefilling_queue.end();) {
            if (*it == seq_id) {
                it = prefilling_queue.erase(it);
            } else {
                ++it;
            }
        }
    }
}

ErrorCode Scheduler::moveDecodingToFinished(const Batch& decode_batch) {
    for (size_t seq_id : decode_batch.sequence_ids) {
        auto seq = seq_pool->get(seq_id);
        if (!seq) {
            continue;
        }
        if (seq->state == SequenceState::DECODING) {
            if (seq->token_ids.size() >= engine_config.max_sequence_length
                || seq->token_ids.size() >= seq->seq_config.max_tokens
                || seq->token_ids.back() == engine_config.model_config.eos_token_id) {
                seq->state = SequenceState::FINISHED;
                LOG_DEBUG("Sequence " + std::to_string(seq->seq_id) + " moved to FINISHED state.");
            }
        }
    }
    return ErrorCode::SUCCESS;
}

void Scheduler::appendDecodedTokens(Batch& decode_batch) {
    if (decode_batch.sampled_token_ids.size() != decode_batch.sequence_ids.size()) {
        return;
    }

    for (size_t i = 0; i < decode_batch.sequence_ids.size(); ++i) {
        auto seq = seq_pool->get(decode_batch.sequence_ids[i]);
        if (!seq) {
            continue;
        }
        const size_t current_time = current_time_ns();

        if (seq->generated_token_count == 0) {
            seq->first_token_time = current_time;
        } else {
            seq->itl_sum += (current_time - seq->last_token_time);
            seq->itl_count += 1;
        }

        seq->last_token_time = current_time;
        seq->generated_token_count += 1;
        seq->add_token(decode_batch.sampled_token_ids[i]);
    }
}

ErrorCode Scheduler::movePrefilledToDecoding(const Batch& prefill_batch) {
    std::lock_guard<std::mutex> lock(queue_mutex);
    std::unordered_set<size_t> moved;
    for (size_t seq_id : prefill_batch.sequence_ids) {
        if (!moved.insert(seq_id).second) {
            continue;
        }
        std::shared_ptr<Sequence> seq = seq_pool->get(seq_id);
        if (!seq) {
            continue;
        }
        if (seq->state == SequenceState::PREFILLED) {
            seq->state = SequenceState::DECODING;
            LOG_DEBUG("Sequence " + std::to_string(seq->seq_id) + " moved to DECODING state.");
            decoding_queue.push_back(seq_id);

            for (auto qit = prefilling_queue.begin(); qit != prefilling_queue.end(); ++qit) {
                if (*qit == seq_id) {
                    prefilling_queue.erase(qit);
                    break;
                }
            }
        }
    }
    return ErrorCode::SUCCESS;
}

std::variant<Batch, ErrorCode> Scheduler::buildDecodeBatch() {
    std::lock_guard<std::mutex> lock(queue_mutex);
    Batch batch;
    batch.batch_id = next_batch_id.fetch_add(1);
    batch.batch_size = 0;
    batch.num_tokens = 0;

    for (size_t seq_id : decoding_queue) {
        auto seq = seq_pool->get(seq_id);
        if (!seq) {
            continue;
        }
        if (seq->state == SequenceState::FINISHED || seq->token_ids.empty()) {
            continue;
        }

        size_t last_pos = seq->token_ids.size() - 1;
        batch.token_ids.push_back(seq->token_ids[last_pos]);
        batch.token_positions.push_back(last_pos);
        batch.sequence_ids.push_back(seq_id);
        batch.num_tokens += 1;
        batch.batch_size++;

        if (batch.batch_size >= engine_config.max_decode_batch_size) {
            break;
        }
    }

    while (batch.batch_size < engine_config.max_decode_batch_size && !prefilling_queue.empty()) {
        size_t seq_id = prefilling_queue.front();
        auto seq = seq_pool->get(seq_id);
        prefilling_queue.erase(prefilling_queue.begin());
        if (!seq) {
            continue;
        }
        if (seq->state != SequenceState::PREFILLED) {
            continue;
        }
        seq->state = SequenceState::DECODING;
        decoding_queue.push_back(seq_id);
        LOG_DEBUG("Sequence " + std::to_string(seq->seq_id) + " moved from PREFILLED to DECODING state.");

        if (seq->token_ids.empty()) {
            continue;
        }

        size_t last_pos = seq->token_ids.size() - 1;
        batch.token_ids.push_back(seq->token_ids[last_pos]);
        batch.token_positions.push_back(last_pos);
        batch.sequence_ids.push_back(seq_id);
        batch.num_tokens += 1;
        batch.batch_size++;
    }

    LOG_DEBUG("BUILD DECODE BATCH: batch_size=" + std::to_string(batch.batch_size) + ", num_tokens=" + std::to_string(batch.num_tokens));
    return batch;
}

std::variant<Batch, ErrorCode> Scheduler::buildPrefillBatch() {
    std::lock_guard<std::mutex> lock(queue_mutex);
    Batch batch;
    batch.batch_id = next_batch_id.fetch_add(1);
    batch.batch_size = 0;
    batch.num_tokens = 0;

    for (auto it = waiting_queue.begin(); it != waiting_queue.end();) {
        size_t seq_id = *it;
        auto seq = seq_pool->get(seq_id);
        if (!seq) {
            it = waiting_queue.erase(it);
            continue;
        }
        if (seq->state != SequenceState::WAITING) {
            ++it;
            continue;
        }

        it = waiting_queue.erase(it);
        seq->state = SequenceState::PREFILLING;
        prefilling_queue.push_back(seq_id);
        LOG_DEBUG("Sequence " + std::to_string(seq->seq_id) + " moved to PREFILLING state.");

        batch.token_ids.insert(batch.token_ids.end(), seq->token_ids.begin(), seq->token_ids.end());
        size_t seq_len = seq->token_ids.size();
        for (size_t i = 0; i < seq_len; ++i) {
            batch.token_positions.push_back(i);
            batch.sequence_ids.push_back(seq_id);
        }
        batch.max_token_positions.push_back(seq_len - 1);
        batch.num_tokens += seq_len;
        batch.batch_size++;

        LOG_DEBUG("Sequence " + std::to_string(seq->seq_id) + " added to prefill batch with " + std::to_string(seq_len) + " tokens.");

        if (batch.batch_size >= engine_config.max_prefill_batch_size) {
            break;
        }
    }

    LOG_DEBUG("BUILD PREFILL BATCH: batch_size=" + std::to_string(batch.batch_size) + ", num_tokens=" + std::to_string(batch.num_tokens));
    return batch;
}

ErrorCode Scheduler::addSequence(
    size_t seq_id,
    std::vector<size_t> token_ids,
    const SequenceConfig& sequence_config
) {
    auto new_seq = seq_pool->create(seq_id, sequence_config);
    new_seq->token_ids = token_ids;
    new_seq->seq_len = token_ids.size();
    new_seq->state = SequenceState::PREPARED;
    new_seq->blocks.clear();

    std::lock_guard<std::mutex> lock(queue_mutex);
    prepared_queue.push_back(seq_id);
    new_seq->submitted_time = current_time_ns();

    LOG_DEBUG("Sequence added to prepared queue: " + std::to_string(new_seq->seq_id));
    queue_cv.notify_one();
    return ErrorCode::SUCCESS;
}

ErrorCode Scheduler::launchSequence() {
    std::lock_guard<std::mutex> lock(queue_mutex);
    while (!prepared_queue.empty()) {
        size_t seq_id = prepared_queue.front();
        prepared_queue.erase(prepared_queue.begin());
        auto seq = seq_pool->get(seq_id);
        if (!seq) {
            continue;
        }
        seq->state = SequenceState::WAITING;
        waiting_queue.push_back(seq_id);
        LOG_DEBUG("Sequence " + std::to_string(seq->seq_id) + " moved to WAITING state.");
    }
    return ErrorCode::SUCCESS;
}

ErrorCode Scheduler::handleFinishedSequence() {
    std::lock_guard<std::mutex> lock(queue_mutex);
    for (auto it = decoding_queue.begin(); it != decoding_queue.end();) {
        auto seq = seq_pool->get(*it);
        if (seq && seq->state == SequenceState::FINISHED) {
            finished_queue.push_back(*it);
            LOG_DEBUG("Sequence " + std::to_string(seq->seq_id) + " moved to FINISHED queue.");
            it = decoding_queue.erase(it);
        } else {
            ++it;
        }
    }
    return ErrorCode::SUCCESS;
}

ErrorCode Scheduler::returnSequenceOutput() {
    std::vector<size_t> finished_to_free;
    {
        std::lock_guard<std::mutex> queue_lock(queue_mutex);
        for (size_t seq_id : finished_queue) {
            auto seq = seq_pool->get(seq_id);
            if (!seq) {
                continue;
            }
            //notify waiting get_request_output to return output and free sequence
            if (seq->state == SequenceState::FINISHED && !seq->finish_handled) {
                std::lock_guard<std::mutex> lock(seq->mtx);
                seq->cv.notify_one();
                seq->finish_handled = true;
                finished_to_free.push_back(seq_id);
            }
        }
    }
    //notify workers to free cache for finished sequences
    if (!finished_to_free.empty()) {
        freeFinishedSequencesOnWorkers(finished_to_free);
    }
    return ErrorCode::SUCCESS;
}

void Scheduler::freeFinishedSequencesOnWorkers(const std::vector<size_t>& sequence_ids) {
    if (sequence_ids.empty()) {
        return;
    }
    //
    CoordinatorExecutor* coordinator = dynamic_cast<CoordinatorExecutor*>(model_executor.get());
    if(!coordinator) {
        LOG_ERROR("Scheduler's model executor is not a CoordinatorExecutor");
        return;
    }
    //build a lightwight control batch with sequence ids 
    // to notify workers to free cache for these finished sequences
    Batch control_batch;
    control_batch.sequence_ids = sequence_ids;
    coordinator->run_free(control_batch);
}

ErrorCode Scheduler::getSequenceById(size_t seq_id, std::shared_ptr<Sequence>& seq) {
    seq = seq_pool->get(seq_id);
    if (!seq) {
        return ErrorCode::SEQUENCE_NOT_FOUND;
    }
    return ErrorCode::SUCCESS;
}

ErrorCode Scheduler::getFinishedSequenceById(size_t seq_id, std::shared_ptr<Sequence>& seq) {
    seq = nullptr;
    std::lock_guard<std::mutex> lock(queue_mutex);
    for (size_t finished_id : finished_queue) {
        if (finished_id == seq_id) {
            seq = seq_pool->get(seq_id);
            return seq ? ErrorCode::SUCCESS : ErrorCode::SEQUENCE_NOT_FOUND;
        }
    }
    return ErrorCode::SEQUENCE_NOT_FOUND;
}

ErrorCode Scheduler::removeFinishedSequenceById(size_t seq_id) {
    std::lock_guard<std::mutex> lock(queue_mutex);
    for (auto it = finished_queue.begin(); it != finished_queue.end(); ++it) {
        if (*it == seq_id) {
            finished_queue.erase(it);
            seq_pool->erase(seq_id);
            LOG_DEBUG("Sequence " + std::to_string(seq_id) + " removed from finished queue and seq pool.");
            return ErrorCode::SUCCESS;
        }
    }
    return ErrorCode::SEQUENCE_NOT_FOUND;
}
