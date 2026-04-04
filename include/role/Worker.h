#pragma once
#include "llm_engine_config.h"
#include "role/Role.h"
#include "executor/Executor.h"
#include "executor/SingleCardExecutor.h"
#include "executor/PiplineExecutor.h"
#include "model/IModel.h"
#include "Workspace.h"
#include "KVCacheManager.h"
#include "error.h"
#include "channel/Channel.h"
#include "channel/ChannelMessage.h"
#include "SequencePool.h"
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
                if(engine_config.enable_pipeline_parallel){
                    model_executor = std::make_unique<PiplineExecutor>(
                        model, 
                        workspace, 
                        engine_config.stage_start_layer, 
                        engine_config.stage_end_layer,
                        seq_pool.get()
                    );
                } else {
                    model_executor = std::make_unique<SingleCardExecutor>(
                        model, 
                        workspace,
                        seq_pool.get()
                    );
                }
                this->cache_manager = cache_manager;
               
            }


        void run() override;
        void work();
        void request_stop();
        ErrorCode receive(ForwardMessage& message);
        ErrorCode send(const ForwardMessage& message);

        void set_channels();

    private:
        KVCacheManager* cache_manager;
        std::unique_ptr<SequencePool> seq_pool;
        std::unique_ptr<Executor> model_executor;
        LLMEngineConfig engine_config;

        Channel* from_scheduler = nullptr;
        Channel* to_scheduler = nullptr;
        Channel* from_prev_worker = nullptr;
        Channel* to_next_worker = nullptr;

        std::unordered_map<size_t, cudaEvent_t> retained_outgoing_events;
    std::atomic<bool> stop_requested{false};

        void freeFinishedSequencesOnWorkers(const std::vector<size_t>& sequence_ids);
        void cleanup_retained_events();

};