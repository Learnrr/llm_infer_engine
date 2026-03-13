#pragma once

#include"Sequence.h"
#include"define.h"
#include "KVCacheManager.h"
#include "model.h"
#include "Batch.h"
#include "CacheBlock.h"
#include <thread>
#include <vector>
#include "utils/include/error.h"
#include <variant>

class Scheduler{
    public:
        Scheduler(KVCacheManager* cache_manager, Model* model) {
            init(cache_manager, model);
        }
        ErrorCode init(KVCacheManager* cache_manager, Model* model) {
            this->cache_manager = cache_manager;
            this->model = model;
            return ErrorCode::SUCCESS;
        }

        void schedule();
        
        ErrorCode movePrefilledToDecoding(const Batch& prefill_batch);

        variant<Batch, ErrorCode> buildDecodeBatch();

        variant<Batch, ErrorCode> buildPrefillBatch();
        ErrorCode addSequence(size_t seq_id, vector<size_t> token_ids);


        ErrorCode launchSequence();

        ErrorCode handleFinishedSequence();

        ErrorCode getSequenceById(size_t seq_id, Sequence* seq);

        ErrorCode getFinishedSequenceById(size_t seq_id, Sequence* seq);

        ErrorCode returnSequenceOutput();

        ErrorCode removeFinishedSequenceById(size_t seq_id);

    private:
        vector<shared_ptr<Sequence>> waiting_queue;
        vector<shared_ptr<Sequence>> decoding_queue;
        vector<shared_ptr<Sequence>> prefilling_queue;
        vector<shared_ptr<Sequence>> finished_queue;

        KVCacheManager* cache_manager;
        Model* model;
        
}