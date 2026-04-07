#pragma once
#include "llm_engine_config.h"
#include "role/Role.h"
#include "executor/Executor.h"
#include "executor/SingleCardExecutor.h"
#include "executor/PipelineExecutor.h"
#include "model/IModel.h"
#include "Workspace.h"
#include "KVCacheManager.h"
#include "error.h"
#include "channel/Channel.h"
#include "channel/ChannelMessage.h"
#include "SequencePool.h"
#include "PrefixCacheManager.h"
#include <unordered_map>
#include <atomic>
class Worker: public Role {
    public:
        Worker(
            KVCacheManager* cache_manager,
            IModel* model,
            Workspace* workspace,
            const LLMEngineConfig& engine_config
        ){
                this->engine_config = engine_config;
                this->seq_pool = std::make_unique<SequencePool>();
                //prefix caching
                if(engine_config.enable_prefix_cache){
                    prefix_cache_manager = std::make_unique<PrefixCacheManager>(engine_config);
                } else {
                    prefix_cache_manager = nullptr;
                }
                      
                if(engine_config.enable_pipeline_parallel){
                    model_executor = std::make_unique<PipelineExecutor>(
                        model, 
                        workspace, 
                        engine_config.stage_start_layer, 
                        engine_config.stage_end_layer,
                        seq_pool.get(),
                        cache_manager,
                        prefix_cache_manager.get(),
                        &retained_outgoing_events
                    );
                } else {
                    model_executor = std::make_unique<SingleCardExecutor>(
                        model, 
                        workspace,
                        seq_pool.get(),
                        cache_manager,
                        prefix_cache_manager.get(),
                        &retained_outgoing_events
                    );
                }
                this->cache_manager = cache_manager;
                this->workspace = workspace;
            } 


        void run() override;
        void work();
        ErrorCode receive(ForwardMessage& message);
        ErrorCode send(const ForwardMessage& message);

        void set_channels();

    private:
        KVCacheManager* cache_manager;
        std::unique_ptr<SequencePool> seq_pool;
        std::unique_ptr<Executor> model_executor;
        LLMEngineConfig engine_config;
        Workspace* workspace;
        std::unique_ptr<PrefixCacheManager> prefix_cache_manager;

        // Communication channels
        Channel* from_scheduler = nullptr;
        Channel* to_scheduler = nullptr;
        Channel* from_prev_worker = nullptr;
        Channel* to_next_worker = nullptr;

        std::unordered_map<size_t, cudaEvent_t> retained_outgoing_events;
        std::atomic<bool> stop_requested{false};

        void cleanup_retained_events();
        void setdevice();
        ErrorCode allocate_blocks(ForwardMessage& message);
        ErrorCode handle_remote_forward(ForwardMessage& message, void** external_hidden_out);
        ErrorCode handle_local_forward(ForwardMessage& message);
        ErrorCode build_response_and_send(ForwardMessage& message, void* external_hidden_out, size_t produced_hidden_tokens);
        ErrorCode bind_cacheblocks_for_batch(const Batch& batch);
        ErrorCode trim_prefill_batch_after_prefix_bind(Batch& batch);

};