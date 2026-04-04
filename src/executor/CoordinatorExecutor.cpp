#include "executor/CoordinatorExecutor.h"

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
    if (response.op_type != ForwardOp::DONE) {
        LOG_ERROR(
            "CoordinatorExecutor expected DONE response from worker, got " +
            std::to_string(static_cast<int>(response.op_type))
        );
        return false;
    }
    if (!response.batch.sampled_token_ids.empty()) {
        batch.sampled_token_ids = response.batch.sampled_token_ids;
    }
    return true;
}
} // namespace




void CoordinatorExecutor::run_prefill(Batch& batch) {
    dispatch_to_worker(to_worker0, ForwardOp::PREFILL, batch);
    last_forward_ok = receive_from_worker(from_worker_last, batch);
}

void CoordinatorExecutor::run_decode(Batch& batch) {
    dispatch_to_worker(to_worker0, ForwardOp::DECODE, batch);
    last_forward_ok = receive_from_worker(from_worker_last, batch);
}

void CoordinatorExecutor::run_free(Batch& batch) {
    dispatch_to_worker(to_worker0, ForwardOp::FREE_SEQ, batch);

    ForwardMessage response;
    from_worker_last->receive(response);
    if (response.op_type != ForwardOp::DONE) {
        LOG_ERROR("CoordinatorExecutor expected DONE response for FREE_SEQ");
        return;
    }
}

void CoordinatorExecutor::run_release_events(Batch& batch) {
    dispatch_to_worker(to_worker0, ForwardOp::RELEASE_EVENTS, batch);

    ForwardMessage response;
    from_worker_last->receive(response);
    if (response.op_type == ForwardOp::DONE) {
        return;
    }
    if (response.op_type == ForwardOp::RELEASE_EVENTS_FAILED) {
        LOG_ERROR("CoordinatorExecutor received RELEASE_EVENTS_FAILED for batch_id=" + std::to_string(batch.batch_id));
        return;
    }
    {
        LOG_ERROR("CoordinatorExecutor expected DONE/RELEASE_EVENTS_FAILED response for RELEASE_EVENTS");
        return;
    }
}

void CoordinatorExecutor::run_stop() {
    Batch control_batch;
    dispatch_to_worker(to_worker0, ForwardOp::STOP, control_batch);

    if (from_worker_last == nullptr) {
        LOG_ERROR("CoordinatorExecutor input channel is null for STOP");
        return;
    }

    ForwardMessage response;
    from_worker_last->receive(response);
    if (response.op_type != ForwardOp::DONE) {
        LOG_ERROR("CoordinatorExecutor expected DONE response for STOP");
    }
}

bool CoordinatorExecutor::consume_last_forward_ok() {
    const bool ok = last_forward_ok;
    last_forward_ok = true;
    return ok;
}