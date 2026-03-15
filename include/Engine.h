#pragma once

#include "KVCacheManager.h"
#include "Workspace.h"
#include "Sequence.h"
#include "IModel.h"
#include "Tensor.h"
#include "define.h"
#include <vector>
#include <thread>
#include "utils/include/error.h"
#include "ModelConfig.h"
#include "llm_engine_config.h"
class Engine{
    public:
        Engine(const Engine&) = delete;
        Engine& operator=(const Engine&) = delete;  
        ~Engine() {
            if (scheduler) {
                scheduler->request_stop();
            }
            if (runner_thread.joinable()) {
                runner_thread.join();
            }
        }

        static Engine* get_instance() {
            static Engine instance;
            return &instance;
        }

        void init(char* llm_engine_config_path);

        void run(); 

        void create_sequence(size_t seq_id, vector<size_t> token_ids);

        void get_sequence_output(size_t seq_id, vector<size_t>& output_token_ids);

        void check_sequence_state(size_t seq_id, SequenceState& state);



    private:
        Engine() = default;
        std::unique_ptr<IModel> model;
        std::unique_ptr<Scheduler> scheduler;
        std::unique_ptr<KVCacheManager> cache_manager;
        std::unique_ptr<Workspace> workspace;

        LLMEngineConfig engine_config;
        std::thread runner_thread;


    };