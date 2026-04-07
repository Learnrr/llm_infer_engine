#pragma once

#include "executor/Executor.h"
#include "channel/Channel.h"
#include "channel/ChannelMessage.h"
#include "model/IModel.h"
#include "Batch.h"
#include <deque>
#include <mutex>

class CoordinatorExecutor : public Executor {
public:
    explicit CoordinatorExecutor(IModel* model = nullptr) : model(model) {}

    void set_channels(
        Channel* to_worker0,
        Channel* from_worker_last
    ) override {
        this->to_worker0 = to_worker0;
        this->from_worker_last = from_worker_last;
    }

    ErrorCode run_prefill(Batch& batch, ModelForwardContext& context) override;
    ErrorCode run_decode(Batch& batch, ModelForwardContext& context) override;
    ErrorCode run_free(Batch& batch) override;
    ErrorCode run_release_events(Batch& batch) override;
    ErrorCode run_stop() override;
    bool consume_last_forward_ok();

    bool poll_completion(CompletionRecord& out_record) override;

    ErrorCode run_prefix_probe(Batch& batch) override;

private:
    void push_completion(CompletionRecord record);

    IModel* model = nullptr;
    Channel* to_worker0 = nullptr;
    Channel* from_worker_last = nullptr;
    bool last_forward_ok = true;
    std::deque<CompletionRecord> completed_records;
    std::mutex completion_mutex;

};