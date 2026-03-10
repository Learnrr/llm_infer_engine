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
            
        }

        static Engine* get_instance() {
            if (instance == nullptr) {
                instance = new Engine();
            }
            return instance;
        }

        void init(){
            cache_manager = make_unique<KVCacheManager>();
            workspace = make_unique<Workspace>();
            model = make_unique<Model>(*workspace);
            scheduler = make_unique<Scheduler>(cache_manager, model);
        }
        
        void

        void run() {
            scheduler->schedule();
        }   

        void create_sequence(size_t seq_id, vector<size_t> token_ids) {
            scheduler->addSequence(seq_id, token_ids);
        }

    private:
        Engine() {
            init();
        }
        static Engine* instance;
        std::unique_ptr<Model> model;
        std::unique_ptr<Scheduler> scheduler;
        std::unique_ptr<KVCacheManager> cache_manager;
        std::unique_ptr<Workspace> workspace;
}