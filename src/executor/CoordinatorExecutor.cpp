#include "executor/CoordinatorExecutor.h"
#include <algorithm>
#include "utils/logger.h"

namespace {
void dispatch_to_worker(
    Channel* output,
    ForwardOp op,
    Batch& batch
) {
    if (output == nullptr) {
        LOG_ERROR("CoordinatorExecutor output channel is null");
        return;
    }

    ForwardMessage message;
    message.op_type = op;
    message.batch = batch;
    output->send(message);
}

bool receive_from_worker(Channel* input, Batch& batch) {
    if (input == nullptr) {
        LOG_ERROR("CoordinatorExecutor input channel is null.");
        return false;
    }   

    ForwardMessage response;
    input->receive(response);
    if (response.op_type == ForwardOp::INVALID) {
        LOG_ERROR(
            "Coordinator received INVALID response from worker for batch_id=" +
            std::to_string(batch.batch_id)
        );
        return false;
    }
    if (response.op_type == ForwardOp::RELEASE_EVENTS_FAILED) {
        LOG_ERROR(
            "Coordinator received RELEASE_EVENTS_FAILED from worker for batch_id=" +
            std::to_string(batch.batch_id)
        );
        return false;
    }
    //handle prefix probe response
    if(response.op_type == ForwardOp::PREFIX_PROBE_RESPONSE){
         if(response.batch.prefix_hit_tokens_per_seq.empty()){
            LOG_ERROR("Coordinator received empty PREFIX_PROBE_RESPONSE from worker for batch_id=" + std::to_string(batch.batch_id));
            return false;
        }
        batch.prefix_hit_tokens_per_seq = std::move(response.batch.prefix_hit_tokens_per_seq);
        return true;
    }
    if (response.op_type != ForwardOp::DONE) {
        LOG_ERROR(
            "Coordinator expected DONE response from worker, got " +
            std::to_string(static_cast<int>(response.op_type))
        );
        return false;
    }
    
    if (!response.batch.sampled_token_ids.empty()) {
        batch.sampled_token_ids = std::move(response.batch.sampled_token_ids);
    }
    return true;
}
} // namespace




ErrorCode CoordinatorExecutor::run_prefill(Batch& batch, ModelForwardContext& context) {
    (void)context;
    dispatch_to_worker(to_worker0, ForwardOp::PREFILL, batch);
    last_forward_ok = receive_from_worker(from_worker_last, batch);
    if (!last_forward_ok) {
        LOG_ERROR("Coordinator prefill forward failed; releasing events and reporting error.");
        (void)run_release_events(batch);
        return ErrorCode::UNKNOWN_ERROR;
    }

    if (run_release_events(batch) != ErrorCode::SUCCESS) {
        LOG_ERROR("Coordinator prefill release_events failed.");
        return ErrorCode::UNKNOWN_ERROR;
    }
    CompletionRecord completion;
    completion.batch_id = batch.batch_id;
    completion.op_type = ForwardOp::PREFILL;
    completion.status = CompletionStatus::DONE;
    completion.sequence_ids = batch.sequence_ids;
    completion.sampled_token_ids = batch.sampled_token_ids;
    push_completion(std::move(completion));
    return ErrorCode::SUCCESS;
}

ErrorCode CoordinatorExecutor::run_decode(Batch& batch, ModelForwardContext& context) {
    (void)context;
    dispatch_to_worker(to_worker0, ForwardOp::DECODE, batch);
    last_forward_ok = receive_from_worker(from_worker_last, batch);
    if (!last_forward_ok) {
        LOG_ERROR("Coordinator decode forward failed; releasing events and reporting error.");
        (void)run_release_events(batch);
        return ErrorCode::UNKNOWN_ERROR;
    }

    if (run_release_events(batch) != ErrorCode::SUCCESS) {
        LOG_ERROR("Coordinator decode release_events failed.");
        return ErrorCode::UNKNOWN_ERROR;
    }
    CompletionRecord completion;
    completion.batch_id = batch.batch_id;
    completion.op_type = ForwardOp::DECODE;
    completion.status = CompletionStatus::DONE;
    completion.sequence_ids = batch.sequence_ids;
    completion.sampled_token_ids = batch.sampled_token_ids;
    push_completion(std::move(completion));
    return ErrorCode::SUCCESS;
}

ErrorCode CoordinatorExecutor::run_free(Batch& batch) {
    if (to_worker0 == nullptr || from_worker_last == nullptr) {
        LOG_ERROR("CoordinatorExecutor cannot run FREE_SEQ: channel is null");
        last_forward_ok = false;
        return ErrorCode::INITIANLIZATION_ERROR;
    }

    dispatch_to_worker(to_worker0, ForwardOp::FREE_SEQ, batch);
    last_forward_ok = receive_from_worker(from_worker_last, batch);
    return last_forward_ok ? ErrorCode::SUCCESS : ErrorCode::UNKNOWN_ERROR;
}

ErrorCode CoordinatorExecutor::run_release_events(Batch& batch) {
    if (to_worker0 == nullptr || from_worker_last == nullptr) {
        LOG_ERROR("CoordinatorExecutor cannot run RELEASE_EVENTS: channel is null");
        last_forward_ok = false;
        return ErrorCode::INITIANLIZATION_ERROR;
    }

    dispatch_to_worker(to_worker0, ForwardOp::RELEASE_EVENTS, batch);
    last_forward_ok = receive_from_worker(from_worker_last, batch);
    return last_forward_ok ? ErrorCode::SUCCESS : ErrorCode::UNKNOWN_ERROR;
}

ErrorCode CoordinatorExecutor::run_stop() {
    if (to_worker0 == nullptr) {
        LOG_ERROR("CoordinatorExecutor cannot run STOP: output channel is null");
        return ErrorCode::INITIANLIZATION_ERROR;
    }

    Batch control_batch;
    dispatch_to_worker(to_worker0, ForwardOp::STOP, control_batch);
    // best effort stop
    return ErrorCode::SUCCESS;
}

bool CoordinatorExecutor::consume_last_forward_ok() {
    const bool ok = last_forward_ok;
    last_forward_ok = true;
    return ok;
}

bool CoordinatorExecutor::poll_completion(CompletionRecord& out_record) {
    std::lock_guard<std::mutex> lock(completion_mutex);
    if (completed_records.empty()) {
        return false;
    }
    out_record = std::move(completed_records.front());
    completed_records.pop_front();
    return true;
}

void CoordinatorExecutor::push_completion(CompletionRecord record) {
    std::lock_guard<std::mutex> lock(completion_mutex);
    completed_records.push_back(std::move(record));
}

ErrorCode CoordinatorExecutor::run_prefix_probe(Batch& batch){
    if (to_worker0 == nullptr || from_worker_last == nullptr) {
        LOG_ERROR("CoordinatorExecutor cannot run PREFIX_PROBE: channel is null");
        last_forward_ok = false;
        return ErrorCode::INITIANLIZATION_ERROR;
    }

    dispatch_to_worker(to_worker0, ForwardOp::PREFIX_PROBE, batch);
    last_forward_ok = receive_from_worker(from_worker_last, batch);
    if (!last_forward_ok) {
        LOG_ERROR("Coordinator prefix probe failed for batch_id=" + std::to_string(batch.batch_id));
        return ErrorCode::UNKNOWN_ERROR;
    }
    return ErrorCode::SUCCESS;
}