#pragma once

#include "KVCacheManager.h"
#include "RequestManager.h"
#include "metrics/MetricCalculator.h"
#include "Workspace.h"
#include "Scheduler.h"
#include "Worker.h"
#include "Sequence.h"
#include "IModel.h"
#include "Tensor.h"
#include "define.h"
#include <vector>
#include <thread>
#include "error.h"
#include "ModelConfig.h"
#include "llm_engine_config.h"
#include "SequenceOutput.h"
class Engine{
    public:
        Engine(const Engine&) = delete;
        Engine& operator=(const Engine&) = delete;  
        ~Engine() {
            if (scheduler) {
                scheduler->request_stop();
            }
            if (worker) {
                worker->request_stop();
            }
            if (runner_thread.joinable()) {
                runner_thread.join();
            }
        }

        static Engine* get_instance() {
            static Engine instance;
            return &instance;
        }

        //build configs and initialize members
        void init(char* llm_engine_config_path);
        //start the scheduler thread or worker thread
        void run(); 


        //=============scheduler side functions==============
        //public API to submit tokens with sequence config, and get request output
        void submit_tokens(std::vector<size_t> token_ids, size_t& request_id);
        void submit_tokens(std::vector<size_t> token_ids, const SequenceConfig& sequence_config, size_t& request_id);
        //public API to get request output
        void get_request_output(size_t request_id, SequenceOutput& output);
        //public API to check request state
        void check_request_state(size_t request_id, RequestStatus& state);





    private:
        Engine() = default;
        std::unique_ptr<IModel> model;
        std::unique_ptr<Scheduler> scheduler;
        std::unique_ptr<Worker> worker;

        std::unique_ptr<KVCacheManager> cache_manager;
        std::unique_ptr<Workspace> workspace;
        std::unique_ptr<RequestManager> request_manager;
        std::unique_ptr<MetricCalculator> metric_calculator;

        LLMEngineConfig engine_config;
        std::thread runner_thread;
        //functions to create request and submit request
        void create_request(std::vector<size_t> token_ids, size_t& request_id);
        void submit_request(size_t request_id, const SequenceConfig& sequence_config);
        void attach_channel();


    };