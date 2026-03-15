#pragma once

#include"Sequence.h"
#include"define.h"
#include "KVCacheManager.h"
#include "model/include/IModel.h"
#include "Batch.h"
#include "CacheBlock.h"
#include <thread>
#include <vector>
#include <atomic>
#include "utils/include/error.h"
#include <variant>
#include "llm_engine_config.h"

class Scheduler{
    public:
        Scheduler(
            KVCacheManager* cache_manager, 
            IModel* model, 
            Workspace* workspace, 
            const LLMEngineConfig& engine_config
        ) {
            init(cache_manager, model, workspace, engine_config);
        }


        void schedule();
        void request_stop();
    
        ErrorCode addSequence(size_t seq_id, vector<size_t> token_ids);

        ErrorCode getSequenceById(size_t seq_id, std::shared_ptr<Sequence>& seq);

        ErrorCode getFinishedSequenceById(size_t seq_id, std::shared_ptr<Sequence>& seq);

        ErrorCode returnSequenceOutput();

        ErrorCode removeFinishedSequenceById(size_t seq_id);

    private:
        vector<shared_ptr<Sequence>> prepared_queue;
        vector<shared_ptr<Sequence>> waiting_queue;
        vector<shared_ptr<Sequence>> decoding_queue;
        vector<shared_ptr<Sequence>> prefilling_queue;
        vector<shared_ptr<Sequence>> finished_queue;

        KVCacheManager* cache_manager;
        IModel* model;
        LLMEngineConfig engine_config;
        Workspace* workspace;
        size_t eos_token_id;
        std::atomic<bool> stop_requested{false};

        ErrorCode movePrefilledToDecoding(const Batch& prefill_batch);
        ErrorCode moveDecodingToFinished(const Batch& decode_batch);
        variant<Batch, ErrorCode> buildDecodeBatch();
        variant<Batch, ErrorCode> buildPrefillBatch();
        ErrorCode launchSequence();
        ErrorCode handleFinishedSequence();
        void appendDecodedTokens(Batch& decode_batch);
        ErrorCode init(
            KVCacheManager* cache_manager, 
            IModel* model, 
            Workspace* workspace, 
            const LLMEngineConfig& engine_config
        ) {
            this->cache_manager = cache_manager;
            this->model = model;
            this->workspace = workspace;
            this->engine_config = engine_config;
            return ErrorCode::SUCCESS;
        }
};