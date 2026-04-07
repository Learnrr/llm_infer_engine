#pragma once

#include "executor/Executor.h"
#include "model/IModel.h"
#include "SequencePool.h"
#include "Workspace.h"
#include "Batch.h"
#include "model/ModelForwardContext.h"
#include "KVCacheManager.h"
#include <unordered_map>
#include "PrefixCacheManager.h"

class SingleCardExecutor : public Executor {
public:
    SingleCardExecutor(
        IModel* model, 
        Workspace* workspace, 
        SequencePool* seq_pool = nullptr,
        KVCacheManager* cache_manager = nullptr,
        PrefixCacheManager* prefix_cache_manager = nullptr,
        std::unordered_map<size_t, cudaEvent_t>* retained_outgoing_events = nullptr
    )
        : model(model), 
        workspace(workspace), 
        seq_pool(seq_pool),
        cache_manager(cache_manager),
        prefix_cache_manager(prefix_cache_manager),
        retained_outgoing_events(retained_outgoing_events) {}

    ErrorCode run_prefill(Batch& batch, ModelForwardContext& context) override;
    ErrorCode run_decode(Batch& batch, ModelForwardContext& context) override;

    ErrorCode run_release_events(Batch& batch) override;
    ErrorCode run_stop() override;
    ErrorCode run_free(Batch& batch) override;    

    ErrorCode run_prefix_probe(Batch& batch) override;

private:
    IModel* model;
    Workspace* workspace;
    SequencePool* seq_pool;
    KVCacheManager* cache_manager;
    PrefixCacheManager* prefix_cache_manager;

    std::unordered_map<size_t, cudaEvent_t>* retained_outgoing_events;
    ErrorCode write_prefix_to_cache(const Batch& batch);
};