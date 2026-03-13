#include "Engine.h"

void Engine::init(char* model_config_path) {
    cache_manager = std::make_unique<KVCacheManager>();
    ErrorCode error = cache_manager->init();
    if (error != ErrorCode::SUCCESS) {
        // Handle initialization error
        return;
    }

    workspace = std::make_unique<Workspace>();

    ModelConfig config;
    config.build_from_file(model_config_path);

    model = ModelFactory::create_model("QWEN");
    model->init(config);
    model->load_weights(config.model_path.c_str());

    scheduler = std::make_unique<Scheduler>(cache_manager.get(), model.get());
}

void Engine::run() {
    runner_thread = std::thread(&Scheduler::schedule, scheduler.get());
}   

void Engine::create_sequence(size_t seq_id, vector<size_t> token_ids) {
    ErrorCode error = scheduler->addSequence(seq_id, token_ids);
    if (error != ErrorCode::SUCCESS) {
        // Handle error
    }
}

void Engine::get_sequence_output(size_t seq_id, vector<size_t>& output_token_ids) {

    std::shared_ptr<Sequence> seq;
    ErrorCode error = scheduler->getFinishedSequenceById(seq_id, seq);
    if (error != ErrorCode::SUCCESS) {
        // Handle error
        return;
    }
    std::unique_lock<std::mutex> lock(seq->mtx);
    seq->cv.wait(lock, [&seq]{ return seq->state == SequenceState::FINISHED; });
    output_token_ids = seq->output_token_ids;
    //remove the sequence from finished queue
    scheduler->removeFinishedSequenceById(seq_id);
}

void Engine::check_sequence_state(size_t seq_id, SequenceState& state) {
    std::shared_ptr<Sequence> seq;
    ErrorCode error = scheduler->getSequenceById(seq_id, seq);
    if (error != ErrorCode::SUCCESS) {
        // Handle error
        return;
    }
    if (seq) {
        state = seq->state;
    } else {
        //
    }
}