#pragma once

#include"Sequence.h"
#include"define.h"
#include "KVCacheManager.h"
#include "executor/Executor.h"
#include "executor/CoordinatorExecutor.h"
#include "Batch.h"
#include "Cacheblock.h"
#include "SequencePool.h"
#include <thread>
#include <vector>
#include <atomic>
#include <cstdint>
#include <memory>
#include "error.h"
#include <variant>
#include "llm_engine_config.h"
#include <mutex>
#include <condition_variable>
#include "role/Role.h"
#include "channel/Channel.h"

class Scheduler: public Role {
    public:

    Scheduler(
        KVCacheManager* cache_manager,
        IModel* model,
        const LLMEngineConfig& engine_config
    )
        : cache_manager(cache_manager),
        seq_pool(std::make_unique<SequencePool>()),
        model_executor(std::make_unique<CoordinatorExecutor>(model)),
        engine_config(engine_config),
        eos_token_id(engine_config.model_config.eos_token_id) {}



        void run() override;
        void schedule();
        void request_stop();
    
        ErrorCode addSequence(size_t seq_id, std::vector<size_t> token_ids, const SequenceConfig& sequence_config = SequenceConfig());

        ErrorCode getSequenceById(size_t seq_id, std::shared_ptr<Sequence>& seq);

        ErrorCode getFinishedSequenceById(size_t seq_id, std::shared_ptr<Sequence>& seq);

        ErrorCode returnSequenceOutput();

        ErrorCode removeFinishedSequenceById(size_t seq_id);

        void set_channels();

    private:
        std::vector<size_t> prepared_queue;
        std::vector<size_t> waiting_queue;
        std::vector<size_t> decoding_queue;
        std::vector<size_t> prefilling_queue;
        std::vector<size_t> finished_queue;

        std::unique_ptr<SequencePool> seq_pool;

        std::mutex queue_mutex;
        std::condition_variable queue_cv;

        KVCacheManager* cache_manager;
        std::unique_ptr<Executor> model_executor;
        LLMEngineConfig engine_config;
        size_t eos_token_id;
        std::atomic<bool> stop_requested{false};
        std::atomic<uint64_t> next_batch_id{1};

        ErrorCode movePrefilledToDecoding(const Batch& prefill_batch);
        ErrorCode moveDecodingToFinished(const Batch& decode_batch);
        std::variant<Batch, ErrorCode> buildDecodeBatch();
        std::variant<Batch, ErrorCode> buildPrefillBatch();
        ErrorCode launchSequence();
        ErrorCode handleFinishedSequence();
        void appendDecodedTokens(Batch& decode_batch);
        void freeFinishedSequencesOnWorkers(const std::vector<size_t>& sequence_ids);
        void stopWorkers();
        void recoverFromPrefillFailure(const Batch& prefill_batch);
        bool hasPendingWorkLocked() const;
};