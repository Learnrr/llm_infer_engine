#include "Engine.h"
#include<vector>
#include "Scheduler.h"
#include <cstdlib>
#include <string>
#include "utils/logger.h"


void Engine::init(char* llm_engine_config_path) {
    const char* env_log_level = std::getenv("LOG_LEVEL");
    if (env_log_level && env_log_level[0] != '\0') {
        LOG_INFO(std::string("LOG_LEVEL set to ") + env_log_level);
    }
    engine_config.build_from_file(llm_engine_config_path);
    LOG_INFO("Engine config loaded and built from file: " + std::string(llm_engine_config_path));

    request_manager = std::make_unique<RequestManager>();
    LOG_INFO("RequestManager initialized");

    cache_manager = std::make_unique<KVCacheManager>();
    ErrorCode error = cache_manager->init(engine_config);
    if (error != ErrorCode::SUCCESS) {
        // Handle initialization error
        return;
    }
    LOG_INFO("KVCacheManager initialized");

    workspace = std::make_unique<Workspace>();
    error = workspace->init(engine_config);
    if (error != ErrorCode::SUCCESS) {
        // Handle workspace initialization error
        return;
    }
    LOG_INFO("Workspace initialized");


    model = ModelFactory::create_model("QWEN");
    model->init(engine_config);
    model->load_weights(engine_config.model_config.model_path.c_str());
    LOG_INFO("Model weights loaded");


    scheduler = std::make_unique<Scheduler>(
        cache_manager.get(), 
        model.get(), 
        workspace.get(),
        engine_config
    );
    LOG_INFO("Scheduler initialized");
}

void Engine::run() {
    runner_thread = std::thread(&Scheduler::schedule, scheduler.get());
    LOG_INFO("Scheduler started");
}   

//create a request and sequence in manager
void Engine::create_request(std::vector<size_t> token_ids, size_t& request_id){
    ErrorCode error = request_manager->create_request(token_ids, request_id);
    if (error != ErrorCode::SUCCESS) {
        request_id = 0;
    }

}

void Engine::submit_request(size_t request_id) {
    //submit the request to request manager
    ErrorCode error = request_manager->submit_request(request_id);
    if (error != ErrorCode::SUCCESS) {
        return;
    }
    //get the sequence id and token ids corresponding to the request
    size_t seq_id;
    std::vector<size_t> token_ids;
    error = request_manager->get_request_sequence_id(request_id, seq_id);
    if (error != ErrorCode::SUCCESS) {
        request_manager->set_request_status(request_id, RequestStatus::FAILED);
        return;
    }
    error = request_manager->get_request_token_ids(request_id, token_ids);
    if (error != ErrorCode::SUCCESS) {
        request_manager->set_request_status(request_id, RequestStatus::FAILED);
        return;
    }
    //add the sequence corresponding to the request to scheduler
    error = scheduler->addSequence(seq_id, token_ids);
    if (error != ErrorCode::SUCCESS) {
        // Handle error
        request_manager->set_request_status(request_id, RequestStatus::FAILED);
        return;
    }

}

void Engine::get_request_output(size_t request_id, SequenceOutput& output) {
    size_t seq_id;
    ErrorCode error = request_manager->get_request_sequence_id(request_id, seq_id);
    if (error != ErrorCode::SUCCESS) {
        return;
    }

    std::shared_ptr<Sequence> seq;
    error = scheduler->getSequenceById(seq_id, seq);
    if (error != ErrorCode::SUCCESS) {
        // Handle error
        request_manager->set_request_status(request_id, RequestStatus::FAILED);
        return;
    }
    std::unique_lock<std::mutex> lock(seq->mtx);
    seq->cv.wait(lock, [&seq]{ return seq->state == SequenceState::FINISHED; });
    output.seq_id = seq->seq_id;
    output.token_ids = seq->token_ids;
    //remove the sequence from finished queue
    scheduler->removeFinishedSequenceById(seq_id);
    //update request status to completed
    request_manager->set_request_status(request_id, RequestStatus::COMPLETED);
}

void Engine::check_request_state(size_t request_id, RequestStatus& state) {
    ErrorCode error = request_manager->get_request_status(request_id, state);
    if (error != ErrorCode::SUCCESS) {
        state = RequestStatus::FAILED;
    }
}
