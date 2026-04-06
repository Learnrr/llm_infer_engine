#pragma once

#include "executor/Executor.h"
#include "model/IModel.h"
#include "model/ModelForwardContext.h"
#include "SequencePool.h"
#include "Workspace.h"
#include "KVCacheManager.h"
#include <unordered_map>

class PipelineExecutor : public Executor {
public:
    PipelineExecutor(
        IModel* model, 
        Workspace* workspace, 
        size_t stage_start_layer, 
        size_t stage_end_layer, 
        SequencePool* seq_pool = nullptr,
        KVCacheManager* cache_manager = nullptr,
        std::unordered_map<size_t, cudaEvent_t>* retained_outgoing_events = nullptr
    )
        : model(model), 
        workspace(workspace), 
        stage_start_layer(stage_start_layer), 
        stage_end_layer(stage_end_layer), 
        seq_pool(seq_pool),
        cache_manager(cache_manager),
        retained_outgoing_events(retained_outgoing_events) {}

    ErrorCode run_prefill(Batch& batch, ModelForwardContext& context) override;
    ErrorCode run_decode(Batch& batch, ModelForwardContext& context) override;

    ErrorCode run_release_events(Batch& batch) override;
    ErrorCode run_stop() override;
    ErrorCode run_free(Batch& batch) override;       

private:
    IModel* model;
    Workspace* workspace;
    size_t stage_start_layer;
    size_t stage_end_layer;
    SequencePool* seq_pool;
    KVCacheManager* cache_manager;
    std::unordered_map<size_t, cudaEvent_t>* retained_outgoing_events;
};