#include "Engine.h"
#include<vector>
#include "Scheduler.h"
#include <cstdlib>
#include <string>
#include <sstream>
#include "utils/logger.h"
#include <cuda_runtime.h>
#include "channel/ChannelManager.h"
#include "channel/Channel.h"


void Engine::init(char* llm_engine_config_path) {
    const char* env_log_level = std::getenv("LOG_LEVEL");
    if (env_log_level && env_log_level[0] != '\0') {
        LOG_INFO(std::string("LOG_LEVEL set to ") + env_log_level);
    }
    ErrorCode error = engine_config.build_from_file(llm_engine_config_path);
    if (error != ErrorCode::SUCCESS) {
        LOG_ERROR("Failed to build engine config from file: " + std::string(llm_engine_config_path));
        return;
    }
    LOG_INFO("Engine config loaded and built from file: " + std::string(llm_engine_config_path));
    LOG_INFO(
        "EngineConfig - max_decode_batch_size: " + std::to_string(engine_config.max_decode_batch_size) +
        ", max_prefill_batch_size: " + std::to_string(engine_config.max_prefill_batch_size) +
        ", max_sequence_length: " + std::to_string(engine_config.max_sequence_length) +
        ", total_cache_size: " + std::to_string(engine_config.total_cache_size) +
        ", block_size: " + std::to_string(engine_config.block_size) +
        ", temperature: " + std::to_string(engine_config.temperature) +
        ", top_p: " + std::to_string(engine_config.top_p) +
        ", top_k: " + std::to_string(engine_config.top_k) +
        ", model_config_path: " + engine_config.model_config_path
    );    

    // build channels for the pipeline based on the engine configuration.
    ErrorCode channel_error = ChannelManager::get_instance()->build_channels(
        engine_config.role,
        engine_config.world_size,
        engine_config.pipeline_rank
    );
    if (channel_error != ErrorCode::SUCCESS) {
        LOG_ERROR("Failed to build channels for the pipeline.");
        return;
    }
    //scheduler view
    if(engine_config.role == "scheduler"){
        
        request_manager = std::make_unique<RequestManager>();
        LOG_INFO("RequestManager initialized");
        cache_manager = std::make_unique<KVCacheManager>();
        ErrorCode error = cache_manager->init(engine_config);
        if (error != ErrorCode::SUCCESS) {
            // Handle initialization error
            return;
        }
        LOG_INFO("KVCacheManager initialized"); 

        //create model and load weights
        model = ModelFactory::create_model("QWEN");
        model->init(engine_config);
        
        //scheduler
        scheduler = std::make_unique<Scheduler>(
            cache_manager.get(), 
            model.get(),
            engine_config
        );

        //attach communication channels with workers
        attach_channel();
        LOG_INFO("Scheduler initialized");
        metric_calculator = std::make_unique<MetricCalculator>();
        LOG_INFO("MetricCalculator initialized");   

        LOG_INFO("Engine initialized with role: scheduler");
    } else if(engine_config.role == "worker"){
        //set CUDA device for worker only
        cudaError_t set_device_error = cudaSetDevice(engine_config.local_device_id);
        if (set_device_error != cudaSuccess) {
            LOG_ERROR(
                "Failed to set CUDA device to " + std::to_string(engine_config.local_device_id) +
                ": " + std::string(cudaGetErrorString(set_device_error))
            );
            return;
        }
        LOG_INFO("CUDA device set to " + std::to_string(engine_config.local_device_id));

        // build kvcachemanager for worker
        //only worker has kvcachemanager
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

        //create model and load weights
        model = ModelFactory::create_model("QWEN");
        model->init(engine_config);
        model->load_weights(engine_config.model_config.model_path.c_str());
        LOG_INFO("Model weights loaded");   

        //worker setup
        worker = std::make_unique<Worker>(
            cache_manager.get(),
            model.get(),
            workspace.get(),
            engine_config
        );
        LOG_INFO("Worker initialized");        

        //attach communication channels with scheduler and other workers
        attach_channel();
        LOG_INFO("Engine initialized with role: worker");    
    } else {
        LOG_ERROR("Invalid role specified in engine config: " + engine_config.role);
        return;
    }
    
}

void Engine::run() {
    if(engine_config.role == "scheduler"){
        runner_thread = std::thread(&Scheduler::run, scheduler.get());
        LOG_INFO("Scheduler started");
    } else if(engine_config.role == "worker"){
        runner_thread = std::thread(&Worker::run, worker.get());
        LOG_INFO("Worker started");
    } else {
        LOG_ERROR("Invalid role specified in engine config: " + engine_config.role);
        return;
    }
}   

// scheduler side functions
void Engine::submit_tokens(std::vector<size_t> token_ids, const SequenceConfig& sequence_config, size_t& request_id){
    request_id = 0;
    create_request(token_ids, request_id);
    if(request_id == 0){
        LOG_ERROR("Failed to create request for token submission");
        return;
    }
    submit_request(request_id, sequence_config);
}

void Engine::submit_tokens(std::vector<size_t> token_ids, size_t& request_id){
    SequenceConfig default_config;
    default_config.temperature = engine_config.temperature;
    default_config.top_p = engine_config.top_p;
    default_config.top_k = static_cast<int>(engine_config.top_k);
    default_config.max_tokens = engine_config.max_sequence_length;
    submit_tokens(token_ids, default_config, request_id);
}

//create a request and sequence in manager
// allocate a request id and sequence id, and store the token ids in request manager
void Engine::create_request(std::vector<size_t> token_ids, size_t& request_id){
    ErrorCode error = request_manager->create_request(token_ids, request_id);
    if (error != ErrorCode::SUCCESS) {
        request_id = 0;
    }
}
void Engine::submit_request(size_t request_id, const SequenceConfig& sequence_config) {
    //submit the request to request manager
    ErrorCode error = request_manager->submit_request(request_id);
    if (error != ErrorCode::SUCCESS) {
        return;
    }
    //get the sequence id and token ids corresponding to the request
    //at this point only the sequence id is allocated 
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
    //check input errors
    if(token_ids.empty() 
    || token_ids.size() > engine_config.max_sequence_length
    || token_ids.size() > sequence_config.max_tokens
    || sequence_config.max_tokens > engine_config.max_sequence_length
    || sequence_config.max_tokens <=0
    || sequence_config.temperature < 0.0f
    || sequence_config.top_p < 0.0f || sequence_config.top_p > 1.0f
    || sequence_config.top_k <= 0){
        request_manager->set_request_status(request_id, RequestStatus::FAILED);
        return;
    }

    //create the sequence corresponding to the request in scheduler
    //at this point the sequence is created corresponding to the sequence id
    error = scheduler->addSequence(seq_id, token_ids, sequence_config);
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

    //calculate metrics for the sequence
    size_t latency = metric_calculator->calculateLatency(*seq);
    size_t itl = metric_calculator->calculateITL(*seq);
    size_t tpot = metric_calculator->calculateTPOT(*seq);
    size_t ttft = metric_calculator->calculateTTFT(*seq);
    LOG_INFO("Sequence " + std::to_string(seq->seq_id) 
    + " metrics: Latency=" + std::to_string(latency) 
    + "ms, ITL=" + std::to_string(itl) + "ms, TPOT=" 
    + std::to_string(tpot) + "ms" + ", TTFT=" + std::to_string(ttft) + "ms");

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

void Engine::attach_channel() {
    if (engine_config.role == "scheduler" && scheduler) {
        scheduler->set_channels();
        return;
    }

    if (engine_config.role == "worker" && worker) {
        worker->set_channels();
    }
}
