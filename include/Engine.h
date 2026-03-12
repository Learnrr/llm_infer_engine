#pragma once

#include "KVCacheManager.h"
#include "Workspace.h"
#include "Sequence.h"
#include "model.h"
#include "Tensor.h"
#include "define.h"
#include <vector>
class Engine{
    public:
        Engine(const Engine&) = delete;
        Engine& operator=(const Engine&) = delete;  
        ~Engine() {
            delete scheduler;
            delete model;
            delete cache_manager;
            delete workspace;
            
            if (runner_thread.joinable()) {
                runner_thread.join();
            }
        }

        static Engine* get_instance() {
            if (instance == nullptr) {
                instance = new Engine();
            }
            return instance;
        }

        void init();

        void run(); 

        void create_sequence(size_t seq_id, vector<size_t> token_ids);

        void get_sequence_output(size_t seq_id, vector<size_t>& output_token_ids);

        void check_sequence_state(size_t seq_id, SequenceState& state);



    private:
        Engine() {
            init();
        }
        static Engine* instance;
        std::unique_ptr<Model> model;
        std::unique_ptr<Scheduler> scheduler;
        std::unique_ptr<KVCacheManager> cache_manager;
        std::unique_ptr<Workspace> workspace;

        std::thread runner_thread;


}