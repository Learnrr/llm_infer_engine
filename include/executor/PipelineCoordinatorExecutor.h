#pragma once

#include "executor/Executor.h"
#include "channel/Channel.h"
#include "channel/ChannelMessage.h"
#include "model/IModel.h"
#include <deque>
#include <mutex>
#include "Batch.h"
#include "model/ModelForwardContext.h"
#include <thread>
#include <atomic>

class PipelineCoordinatorExecutor : public Executor {
public:
    explicit PipelineCoordinatorExecutor(IModel* model = nullptr) : model(model) {}
    ~PipelineCoordinatorExecutor();

    void set_channels(
        Channel* to_worker0,
        Channel* from_worker_last
    ) override {
        this->to_worker0 = to_worker0;
        this->from_worker_last = from_worker_last;
        start_receive_thread();
    }

    ErrorCode run_prefill(Batch& batch, ModelForwardContext& context) override;
    ErrorCode run_decode(Batch& batch, ModelForwardContext& context) override;
    ErrorCode run_free(Batch& batch) override;
    ErrorCode run_release_events(Batch& batch) override;
    ErrorCode run_stop() override;
    bool consume_last_forward_ok();

    void submit_decode_batch(const Batch& batch);
    void submit_prefill_batch(const Batch& batch);
    bool poll_completion(CompletionRecord& out_record) override;

    void run_prefix_probe(Batch& batch) override;

private:
    void start_receive_thread();
    bool receive_and_track();

    IModel* model = nullptr;
    Channel* to_worker0 = nullptr;
    Channel* from_worker_last = nullptr;
    bool last_forward_ok = true;

    std::deque<CompletionRecord> completed_records;
    std::mutex completion_mutex;
    std::thread receive_thread;
    std::atomic<bool> stop_receive_thread{false};
    std::atomic<bool> receiver_started{false};
};