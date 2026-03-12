#pragma once

#include"Sequence.h"
#include"define.h"
#include "KVCacheManager.h"
#include "model.h"
#include "Batch.h"
#include "CacheBlock.h"
#include <thread>
#include <vector>

class Scheduler{
    public:
        Scheduler(KVCacheManager* cache_manager, Model* model) {
            init(cache_manager, model);
        }
        void init(KVCacheManager* cache_manager, Model* model) {
            this->cache_manager = cache_manager;
            this->model = model;
        }

        void schedule();
        
        void movePrefilledToDecoding(const Batch& prefill_batch);

        Batch buildDecodeBatch();

        Batch buildPrefillBatch();
        void addSequence(size_t seq_id, vector<size_t> token_ids);


        void launchSequence();

        void handleFinishedSequence();

        void getSequenceById(size_t seq_id, Sequence* seq);

        void getFinishedSequenceById(size_t seq_id, Sequence* seq);

    private:
        vector<Sequence> waiting_queue;
        vector<Sequence> decoding_queue;
        vector<Sequence> prefilling_queue;
        vector<Sequence> finished_queue;

        KVCacheManager* cache_manager;
        Model* model;
        
}