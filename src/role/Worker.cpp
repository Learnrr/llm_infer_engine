#include "role/Worker.h"
#include "executor/PipelineExecutor.h"
#include "channel/ChannelManager.h"
#include "utils/logger.h"
#include "Sequence.h"
#include "model/ModelForwardContext.h"
#include <algorithm>
#include <numeric>
#include <unordered_map>

void Worker::set_channels() {
    ChannelManager* manager = ChannelManager::get_instance();
    auto get_or_null = [manager](const std::string& name) -> Channel* {
        Channel* channel = nullptr;
        ErrorCode err = manager->get_channel(name, channel);
        if (err != ErrorCode::SUCCESS) {
            return nullptr;
        }
        return channel;
    };

    const int rank = engine_config.pipeline_rank;
    from_scheduler = engine_config.is_first_stage()
        ? get_or_null("scheduler_to_worker_" + std::to_string(rank))
        : nullptr;
    to_scheduler = engine_config.is_last_stage()
        ? get_or_null("worker_" + std::to_string(rank) + "_to_scheduler")
        : nullptr;

    from_prev_worker = nullptr;
    to_next_worker = nullptr;
    if (rank > 0) {
        from_prev_worker = get_or_null("worker_" + std::to_string(rank - 1) + "_to_worker_" + std::to_string(rank));
    }
    if (rank + 1 < engine_config.world_size) {
        to_next_worker = get_or_null("worker_" + std::to_string(rank) + "_to_worker_" + std::to_string(rank + 1));
    }
}
void Worker::setdevice() {
    cudaError_t set_device_err = cudaSetDevice(engine_config.local_device_id);
    if (set_device_err != cudaSuccess) {
        LOG_ERROR(
            "worker failed to set CUDA device to " + std::to_string(engine_config.local_device_id) +
            ": " + std::string(cudaGetErrorString(set_device_err))
        );
    } else {
        int current_device = -1;
        cudaError_t get_device_err = cudaGetDevice(&current_device);
        if (get_device_err == cudaSuccess) {
            LOG_INFO("worker set CUDA device " + std::to_string(current_device));
        }
    }
}
void Worker::run(){
    setdevice();

    LOG_INFO("worker started running.");
    work();
    cleanup_retained_events();
    LOG_INFO("worker stopped running.");
}
ErrorCode Worker::allocate_blocks(ForwardMessage& message) {
    bool is_prefill = message.op_type == ForwardOp::PREFILL;
    auto& batch = message.batch;

    // allocate KV blocks based on local token positions.
    // prefill may contain multiple tokens per sequence; 
    // decode should have one token per sequence.
    std::unordered_map<size_t, size_t> seq_required_blocks;
    if (is_prefill) {
        for (size_t i = 0; i < batch.num_tokens; ++i) {
            const size_t seq_id = batch.sequence_ids[i];
            const size_t pos = batch.token_positions[i];
            //cal the required block index for the token position for a sequence
            const size_t required_blk_idx = (pos / engine_config.block_size) + 1;
            // keep the max requried block idx for each sequence in the batch
            auto it = seq_required_blocks.find(seq_id);
            if (it == seq_required_blocks.end()) {
                seq_required_blocks[seq_id] = required_blk_idx;
            } else {
                it->second = std::max(it->second, required_blk_idx);
            }
        }
    } else {
        for (size_t i = 0; i < batch.num_tokens; ++i) {
            const size_t seq_id = batch.sequence_ids[i];
            const size_t pos = batch.token_positions[i];
            const size_t required_blk_idx = (pos / engine_config.block_size) + 1;
            seq_required_blocks[seq_id] = required_blk_idx;
        }
    }
    // try to allocate the required blocks for all sequences 
    //in the batch before running the model forward.
    for (const auto& [seq_id, required_blk_idx] : seq_required_blocks) {
        auto seq = seq_pool->get(seq_id);
        if (!seq) {
            seq = seq_pool->create(seq_id);
        }

        auto& blocks = seq->blocks;
        // prefix binding may have already attached some blocks.
        //only allocate missing blocks to reach required block count.
        if (blocks.size() >= required_blk_idx) {
            continue;
        }

        const size_t missing_blocks = required_blk_idx - blocks.size();
        for (size_t i = 0; i < missing_blocks; ++i) {
            auto result = cache_manager->allocate_cache_block();
            if (std::holds_alternative<std::shared_ptr<CacheBlock>>(result)) {
                blocks.push_back(std::get<std::shared_ptr<CacheBlock>>(result));
            } else {
                LOG_ERROR("Worker failed to allocate KV block for sequence " + std::to_string(seq_id));
                return ErrorCode::MEMORY_FAILURE;
            }
        }
    }

    return ErrorCode::SUCCESS;
}

ErrorCode Worker::handle_local_forward(ForwardMessage& message) {
    auto& batch = message.batch;
    ModelForwardContext context;
    context.workspace = workspace;
    context.seq_pool = seq_pool.get();
    if (message.op_type == ForwardOp::PREFILL) {
        model_executor->run_prefill(batch, context);
    } else if (message.op_type == ForwardOp::DECODE) {
        model_executor->run_decode(batch, context);
    }

    return ErrorCode::SUCCESS;
}

ErrorCode Worker::handle_remote_forward(ForwardMessage& message, void** external_hidden_out) {
    // for pipeline, external_hidden_in is hidden states received from previous stage,
    // external_hidden_out is hidden states to be sent to next stage
    // the two pointers all point to this worker's workspace
    void* external_hidden_in = nullptr;
    // remote_base_ptr is the workspace base pointer opened from IPC handle
    void* remote_base_ptr = nullptr;

    void* local_hidden_input = nullptr;  

    Batch& batch = message.batch;
    if (batch.num_tokens == 0) {
        LOG_ERROR("Worker received empty batch message.");
        return ErrorCode::INVALID_INPUT;
    }
    //here we use produced_hidden_tokens from message to detect 
    //if the previous stage produce hidden states with the same token count 
    //as the current batch, 
    // the batch.num_tokens is the trimmed token count in this worker
    // the message.produced_hidden_tokens is the actual token count produced by previous stage
    if (!engine_config.is_first_stage() 
    && message.produced_hidden_tokens > 0 
    && message.produced_hidden_tokens != batch.num_tokens) {
        LOG_ERROR(
            "Worker detected hidden token count mismatch before IPC copy: expected=" +
            std::to_string(batch.num_tokens) + ", produced=" +
            std::to_string(message.produced_hidden_tokens)
        );
        return ErrorCode::INVALID_INPUT;
    }

    if(engine_config.is_first_stage()){
        //build model forward context
        ModelForwardContext context;
        context.workspace = workspace;
        context.seq_pool = seq_pool.get();
        context.start_layer = engine_config.stage_start_layer;
        context.end_layer = engine_config.stage_end_layer;
        context.external_hidden_in = external_hidden_in;
        context.external_hidden_out = external_hidden_out;        
        if(message.op_type == ForwardOp::PREFILL){
            model_executor->run_prefill(batch, context);
        } else {
            model_executor->run_decode(batch, context);
        }
        return ErrorCode::SUCCESS;
    }

    local_hidden_input = workspace->get_hidden_workspace();

    //open remote handle and calculate the address on remote for hidden states input
    cudaIpcMemHandle_t handle = message.cuda_ipc_handle();
    cudaError_t err = cudaIpcOpenMemHandle(&remote_base_ptr, handle, cudaIpcMemLazyEnablePeerAccess);
    if (err != cudaSuccess) {
        LOG_ERROR("Worker failed to open CUDA IPC handle: " + std::string(cudaGetErrorString(err)));
        return ErrorCode::CUDA_FAILURE;
    }
    const size_t incoming_offset = message.cuda_ipc_mem_offset;
    external_hidden_in = static_cast<void*>(static_cast<char*>(remote_base_ptr) + incoming_offset);    

    cudaPointerAttributes ptr_attr{};
    cudaError_t attr_err = cudaPointerGetAttributes(&ptr_attr, external_hidden_in);
    if (attr_err != cudaSuccess) {
        LOG_ERROR("Worker failed to query IPC pointer attributes: " + std::string(cudaGetErrorString(attr_err)));
        cudaIpcCloseMemHandle(remote_base_ptr);
        return ErrorCode::CUDA_FAILURE;
    }    

    // wait cuda event from sender
    cudaEvent_t incoming_ready_event = nullptr;    
    if (message.has_cuda_ipc_event_handle) {
        cudaIpcEventHandle_t event_handle = message.cuda_ipc_event_handle();
        cudaError_t open_event_err = cudaIpcOpenEventHandle(&incoming_ready_event, event_handle);
        if (open_event_err != cudaSuccess) {
            LOG_ERROR("Worker failed to open CUDA IPC event handle: " + std::string(cudaGetErrorString(open_event_err)));
            cudaIpcCloseMemHandle(remote_base_ptr);
            return ErrorCode::CUDA_FAILURE;
        }

        cudaError_t wait_err = cudaStreamWaitEvent(0, incoming_ready_event, 0);
        if (wait_err != cudaSuccess) {
            LOG_ERROR("Worker failed to wait on CUDA IPC event: " + std::string(cudaGetErrorString(wait_err)));
            cudaEventDestroy(incoming_ready_event);
            cudaIpcCloseMemHandle(remote_base_ptr);
            return ErrorCode::CUDA_FAILURE;
        }
    }

    //get local device
    int local_device = -1;
    cudaError_t get_dev_err = cudaGetDevice(&local_device);
    if (get_dev_err != cudaSuccess) {
        LOG_ERROR("Worker failed to query current CUDA device: " + std::string(cudaGetErrorString(get_dev_err)));
        if (incoming_ready_event != nullptr) {
            cudaEventDestroy(incoming_ready_event);
        }
        cudaIpcCloseMemHandle(remote_base_ptr);
        return ErrorCode::CUDA_FAILURE;
    }    
    //calculate hidden state size in bytes
    const size_t hidden_elements = batch.num_tokens * engine_config.model_config.hidden_size;
    const size_t hidden_bytes = hidden_elements * DataTypeBytes(engine_config.model_config.data_type);

    //copy hidden states from imported IPC pointer to local workspace
    cudaError_t copy_err = cudaMemcpyAsync(
        local_hidden_input,
        external_hidden_in,
        hidden_bytes,
        cudaMemcpyDeviceToDevice,
        0
    );

    if (copy_err != cudaSuccess) {
        LOG_ERROR(
            "Worker failed to copy IPC hidden input to local workspace on local device " +
            std::to_string(local_device) + ": " + std::string(cudaGetErrorString(copy_err))
        );
        if (incoming_ready_event != nullptr) {
            cudaEventDestroy(incoming_ready_event);
        }
        cudaIpcCloseMemHandle(remote_base_ptr);
        return ErrorCode::CUDA_FAILURE;
    }
    // synchronize to make sure the copy is done before running the model forward
    cudaError_t copy_sync_err = cudaStreamSynchronize(0);
    if (copy_sync_err != cudaSuccess) {
        LOG_ERROR("Worker failed to synchronize IPC hidden copy on stream 0: " + std::string(cudaGetErrorString(copy_sync_err)));
        if (incoming_ready_event != nullptr) {
            cudaEventDestroy(incoming_ready_event);
        }
        cudaIpcCloseMemHandle(remote_base_ptr);
        return ErrorCode::CUDA_FAILURE;
    }    
    //run forward
    ModelForwardContext context;
    context.workspace = workspace;
    context.seq_pool = seq_pool.get();
    context.start_layer = engine_config.stage_start_layer;
    context.end_layer = engine_config.stage_end_layer;
    context.external_hidden_in = local_hidden_input;
    context.external_hidden_out = engine_config.is_last_stage() ? nullptr : external_hidden_out;

    if (message.op_type == ForwardOp::PREFILL) {
        model_executor->run_prefill(batch, context);
    } else {
        model_executor->run_decode(batch, context);
    }

    // deal with handle and event cleanup after forward
    if (incoming_ready_event != nullptr) {
        cudaError_t destroy_event_err = cudaEventDestroy(incoming_ready_event);
        if (destroy_event_err != cudaSuccess) {
            LOG_ERROR("Worker failed to destroy imported IPC event: " + std::string(cudaGetErrorString(destroy_event_err)));
        }
    }

    cudaError_t close_err = cudaIpcCloseMemHandle(remote_base_ptr);
    if (close_err != cudaSuccess) {
        LOG_ERROR("Worker failed to close CUDA IPC handle: " + std::string(cudaGetErrorString(close_err)));
        return ErrorCode::CUDA_FAILURE;
    }

    return ErrorCode::SUCCESS;
}
// external hidden out is from model forward
ErrorCode Worker::build_response_and_send(
    ForwardMessage& message, 
    void* external_hidden_out,
    size_t produced_hidden_tokens
) {
    //build message for scheduler
    ForwardMessage response;
    const Batch& batch = message.batch;
    if(engine_config.is_last_stage()){
        response.op_type = ForwardOp::DONE;
        response.batch = batch;
        response.produced_hidden_tokens = produced_hidden_tokens;
        // for the last stage worker, already append sampled token ids in model execution
        response.batch.token_ids = batch.token_ids;
        response.batch.sampled_token_ids = batch.sampled_token_ids;
        response.batch.num_tokens = response.batch.token_ids.size();
    } else { //build message for next stage worker
        response.op_type = message.op_type;
        response.batch = batch;
        response.produced_hidden_tokens = produced_hidden_tokens;

        if (external_hidden_out == nullptr) {
            LOG_ERROR("Worker did not produce external_hidden_out for next pipeline stage.");
            return ErrorCode::INVALID_INPUT;
        }
        // build IPC handle from workspace base address plus hidden offset
        void* workspace_base_addr = workspace->get_workspace();
        if (workspace_base_addr == nullptr) {
            LOG_ERROR("Worker workspace base address is invalid when exporting IPC hidden states.");
            return ErrorCode::INVALID_INPUT;
        }

        const size_t out_offset = static_cast<size_t>(
            static_cast<char*>(external_hidden_out) - static_cast<char*>(workspace_base_addr)
        );

        cudaIpcMemHandle_t out_handle{};
        cudaError_t ipc_err = cudaIpcGetMemHandle(&out_handle, workspace_base_addr);
        if (ipc_err != cudaSuccess) {
            LOG_ERROR("Worker failed to get CUDA IPC handle for workspace base address: " + std::string(cudaGetErrorString(ipc_err)));
            return ErrorCode::CUDA_FAILURE;
        }
        // need to create an IPC event to signal the next stage worker when the hidden states are ready
        cudaEvent_t outgoing_ready_event = nullptr;
        cudaError_t create_err = cudaEventCreateWithFlags(
            &outgoing_ready_event,
            cudaEventDisableTiming | cudaEventInterprocess
        );
        if (create_err != cudaSuccess) {
            LOG_ERROR("Worker failed to create IPC event for outgoing hidden states: " + std::string(cudaGetErrorString(create_err)));
            return ErrorCode::CUDA_FAILURE;
        }

        cudaIpcEventHandle_t outgoing_ready_event_handle{};
        cudaError_t export_err = cudaIpcGetEventHandle(&outgoing_ready_event_handle, outgoing_ready_event);
        if (export_err != cudaSuccess) {
            LOG_ERROR("Worker failed to export IPC event handle: " + std::string(cudaGetErrorString(export_err)));
            cudaEventDestroy(outgoing_ready_event);
            return ErrorCode::CUDA_FAILURE;
        }

        cudaError_t record_err = cudaEventRecord(outgoing_ready_event, 0);
        if (record_err != cudaSuccess) {
            LOG_ERROR("Worker failed to record outgoing IPC event: " + std::string(cudaGetErrorString(record_err)));
            cudaEventDestroy(outgoing_ready_event);
            return ErrorCode::CUDA_FAILURE;
        }

        response.set_cuda_ipc_handle(
            out_handle,
            &outgoing_ready_event_handle,
            out_offset
        );

        auto event = retained_outgoing_events.find(batch.batch_id);
        if (event != retained_outgoing_events.end()) {
            LOG_ERROR("Worker already has a retained outgoing event for batch_id=" + std::to_string(batch.batch_id) + ", overwriting it with new event.");
            cudaEventDestroy(event->second);
        }
        retained_outgoing_events[batch.batch_id] = outgoing_ready_event;
    }
    ErrorCode send_error = send(response);
    if (send_error != ErrorCode::SUCCESS) {
        LOG_ERROR("Worker failed to send forward response.");
        return send_error;
    }

    return ErrorCode::SUCCESS;
}
void Worker::work() {
    while (!stop_requested.load()) {
        ForwardMessage message;
        ErrorCode recv_error = receive(message);
        if (recv_error != ErrorCode::SUCCESS) {
            LOG_ERROR("Worker failed to receive forward message.");
            continue;
        }
        //if receive stop signal, send stop to next stage and stop itself
        if (message.op_type == ForwardOp::STOP) {
            ErrorCode stop_error = model_executor->run_stop();

            ForwardMessage stop_response;
            if (stop_error != ErrorCode::SUCCESS) {
                stop_response.op_type = ForwardOp::INVALID;
            } else {
                stop_response.op_type = engine_config.is_last_stage() ? ForwardOp::DONE : ForwardOp::STOP;
            }
            ErrorCode send_error = send(stop_response);
            if (send_error != ErrorCode::SUCCESS) {
                LOG_ERROR("Worker failed to forward STOP control response.");
            }
            stop_requested.store(true);
            break;
        }
        // if receive invalid signal, just forward to next stage and continue to wait for next message
        if (message.op_type == ForwardOp::INVALID) {
            send(message);
            continue;
        }
        // if receive free sequence signal, free the finished sequences on workers 
        // and continue to wait for next message
        if (message.op_type == ForwardOp::FREE_SEQ) {
            ErrorCode free_error = model_executor->run_free(message.batch);

            ForwardMessage cleanup_response;
            cleanup_response.batch.sequence_ids = message.batch.sequence_ids;
            if (free_error != ErrorCode::SUCCESS) {
                cleanup_response.op_type = ForwardOp::INVALID;
            } else if (engine_config.is_last_stage()) {
                cleanup_response.op_type = ForwardOp::DONE;
            } else {
                cleanup_response.op_type = ForwardOp::FREE_SEQ;
            }
            ErrorCode send_error = send(cleanup_response);
            if (send_error != ErrorCode::SUCCESS) {
                LOG_ERROR("Worker failed to forward FREE_SEQ cleanup response.");
            }
            continue;
        }
        if (message.op_type == ForwardOp::RELEASE_EVENTS_FAILED) {
            ForwardMessage failure_response;
            failure_response.batch.batch_id = message.batch.batch_id;
            failure_response.op_type = ForwardOp::RELEASE_EVENTS_FAILED;
            ErrorCode send_error = send(failure_response);
            if (send_error != ErrorCode::SUCCESS) {
                LOG_ERROR("Worker failed to forward RELEASE_EVENTS_FAILED control response.");
            }
            continue;
        }
        // if receive release events, release the retained cuda events for this batch 
        // and send response to next stage 
        if (message.op_type == ForwardOp::RELEASE_EVENTS) {
            ErrorCode release_error = model_executor->run_release_events(message.batch);

            ForwardMessage release_response;
            release_response.batch.batch_id = message.batch.batch_id;
            if (release_error != ErrorCode::SUCCESS) {
                release_response.op_type = ForwardOp::RELEASE_EVENTS_FAILED;
            } else if (engine_config.is_last_stage()) {
                release_response.op_type = ForwardOp::DONE;
            } else {
                release_response.op_type = ForwardOp::RELEASE_EVENTS;
            }
            ErrorCode send_error = send(release_response);
            if (send_error != ErrorCode::SUCCESS) {
                LOG_ERROR("Worker failed to forward RELEASE_EVENTS control response.");
            }
            continue;
        }
        //prefix caching probe handling
        if(message.op_type == ForwardOp::PREFIX_PROBE){
            ErrorCode probe_error = model_executor->run_prefix_probe(message.batch);
            ForwardMessage probe_response;
            if (probe_error != ErrorCode::SUCCESS) {
                probe_response.op_type = ForwardOp::INVALID;
                probe_response.batch.batch_id = message.batch.batch_id;
                probe_response.batch.sequence_ids = message.batch.sequence_ids;
            } else {
                probe_response.op_type = engine_config.is_last_stage() ? ForwardOp::PREFIX_PROBE_RESPONSE : ForwardOp::PREFIX_PROBE;
                probe_response.batch = message.batch;
            }
            ErrorCode send_error = send(probe_response);
            if (send_error != ErrorCode::SUCCESS) {
                LOG_ERROR("Worker failed to forward PREFIX_PROBE_RESPONSE control response.");
            }
            continue;
        }
        //error handling for unknown control message types
        if (message.op_type != ForwardOp::PREFILL && message.op_type != ForwardOp::DECODE) {
            ForwardMessage invalid_response;
            invalid_response.op_type = ForwardOp::INVALID;
            invalid_response.batch.batch_id = message.batch.batch_id;
            invalid_response.batch.sequence_ids = message.batch.sequence_ids;
            ErrorCode send_error = send(invalid_response);
            if (send_error != ErrorCode::SUCCESS) {
                LOG_ERROR("Worker failed to forward INVALID control response.");
            }
            continue;
        }

        // Handle prefill/decode with a compute copy so transport metadata can stay intact.
        // this is important for prefix caching since we need to trim prefill batch after binding
        // hit blocks to sequences
        ForwardMessage forward_message = message;
        Batch& batch = forward_message.batch;

        //handle error case for malformed batch
        batch.num_tokens = batch.num_tokens > 0 ? batch.num_tokens : batch.token_ids.size();
        if (batch.num_tokens == 0) {
            LOG_ERROR("Worker received empty batch message.");
            ForwardMessage invalid_response;
            invalid_response.op_type = ForwardOp::INVALID;
            invalid_response.batch.batch_id = message.batch.batch_id;
            invalid_response.batch.sequence_ids = message.batch.sequence_ids;
            send(invalid_response);
            continue;
        }
        if (batch.sequence_ids.size() < batch.num_tokens || batch.token_positions.size() < batch.num_tokens) {
            LOG_ERROR("Worker received malformed batch: sequence_ids/token_positions size mismatch");
            ForwardMessage invalid_response;
            invalid_response.op_type = ForwardOp::INVALID;
            invalid_response.batch.batch_id = message.batch.batch_id;
            invalid_response.batch.sequence_ids = message.batch.sequence_ids;
            send(invalid_response);
            continue;
        }

        if(engine_config.enable_prefix_cache&& message.op_type == ForwardOp::PREFILL){
            
            //attach hit blocks info for prefix caching
            ErrorCode bind_error = bind_cacheblocks_for_batch(batch);
            if (bind_error != ErrorCode::SUCCESS) {
                ForwardMessage invalid_response;
                invalid_response.op_type = ForwardOp::INVALID;
                invalid_response.batch.batch_id = message.batch.batch_id;
                invalid_response.batch.sequence_ids = message.batch.sequence_ids;
                send(invalid_response);
                continue;
            }

            //remove already hit prefix tokens from prefill compute payload.
            ErrorCode trim_error = trim_prefill_batch_after_prefix_bind(batch);
            if (trim_error != ErrorCode::SUCCESS) {
                ForwardMessage invalid_response;
                invalid_response.op_type = ForwardOp::INVALID;
                invalid_response.batch.batch_id = message.batch.batch_id;
                invalid_response.batch.sequence_ids = message.batch.sequence_ids;
                send(invalid_response);
                continue;
            }

            if(batch.num_tokens == 0){
                // if after trimming prefix hit tokens, there is no more token to prefill, we can directly send a DONE response to next stage or scheduler
                ForwardMessage done_response;
                done_response.op_type = engine_config.is_last_stage() ? ForwardOp::DONE : ForwardOp::PREFILL;
                done_response.batch = message.batch;
                ErrorCode send_error = send(done_response);
                if (send_error != ErrorCode::SUCCESS) {
                    LOG_ERROR("Worker failed to forward DONE response for fully prefix-hit batch.");
                }
                continue;
            }
        }     
       
        //allocate blocks for the batch before forward
        ErrorCode alloc_error = allocate_blocks(forward_message);
        if (alloc_error != ErrorCode::SUCCESS) {
            ForwardMessage invalid_response;
            invalid_response.op_type = ForwardOp::INVALID;
            invalid_response.batch.batch_id = message.batch.batch_id;
            invalid_response.batch.sequence_ids = message.batch.sequence_ids;
            send(invalid_response);
            continue;
        }

        // do model forward
        void* external_hidden_out = nullptr;
        ErrorCode forward_error = ErrorCode::SUCCESS;
        if(!engine_config.enable_pipeline_parallel) {
            forward_error = handle_local_forward(forward_message);
        } else {
            forward_error = handle_remote_forward(forward_message, &external_hidden_out);
        }
        if (forward_error != ErrorCode::SUCCESS) {
            ForwardMessage invalid_response;
            invalid_response.op_type = ForwardOp::INVALID;
            invalid_response.batch.batch_id = message.batch.batch_id;
            invalid_response.batch.sequence_ids = message.batch.sequence_ids;
            send(invalid_response);
            continue;
        }
        // Copy compute outputs from forward batch to transport batch, then send transport batch.
        message.batch.sampled_token_ids = std::move(forward_message.batch.sampled_token_ids);
        ErrorCode send_response_error = build_response_and_send(message, external_hidden_out, forward_message.batch.num_tokens);
        if (send_response_error != ErrorCode::SUCCESS) {
            ForwardMessage invalid_response;
            invalid_response.op_type = ForwardOp::INVALID;
            invalid_response.batch.batch_id = message.batch.batch_id;
            invalid_response.batch.sequence_ids = message.batch.sequence_ids;
            send(invalid_response);
            continue;
        }
    }
}

void Worker::cleanup_retained_events() {
    for (auto& kv : retained_outgoing_events) {
        cudaEvent_t event = kv.second;
        if (event == nullptr) {
            continue;
        }
        cudaError_t destroy_err = cudaEventDestroy(event);
        if (destroy_err != cudaSuccess) {
            LOG_ERROR("Worker failed to destroy retained event during cleanup: " + std::string(cudaGetErrorString(destroy_err)));
        }
    }
    retained_outgoing_events.clear();
}

ErrorCode Worker::receive(ForwardMessage& message) {
    Channel* input = engine_config.is_first_stage() ? from_scheduler : from_prev_worker;
    if (input == nullptr) {
        LOG_ERROR("Worker input channel is null.");
        return ErrorCode::UNKNOWN_ERROR;
    }

    input->receive(message);
    return ErrorCode::SUCCESS;
}

ErrorCode Worker::send(const ForwardMessage& message) {
    Channel* output = engine_config.is_last_stage() ? to_scheduler : to_next_worker;
    if (output == nullptr) {
        LOG_ERROR("Worker output channel is null.");
        return ErrorCode::UNKNOWN_ERROR;
    }

    output->send(message);
    return ErrorCode::SUCCESS;
}


ErrorCode Worker::bind_cacheblocks_for_batch(const Batch& batch){
    if (prefix_cache_manager == nullptr) {
        return ErrorCode::SUCCESS;
    }
    if (batch.prefix_hit_tokens_per_seq.size() != batch.batch_size ||
        batch.max_token_positions.size() != batch.batch_size) {
        LOG_ERROR("Worker received invalid prefix hit metadata for binding.");
        return ErrorCode::INVALID_INPUT;
    }

    size_t cursor = 0;
    for (size_t seq_idx = 0; seq_idx < batch.batch_size; ++seq_idx) {
        const size_t seq_len = batch.max_token_positions[seq_idx] + 1;
        if (cursor + seq_len > batch.sequence_ids.size() ||
            cursor + seq_len > batch.token_ids.size()) {
            LOG_ERROR("Worker prefill batch layout invalid for prefix cache binding.");
            return ErrorCode::INVALID_INPUT;
        }

        const size_t seq_id = batch.sequence_ids[cursor];
        const size_t hit_tokens = std::min(batch.prefix_hit_tokens_per_seq[seq_idx], seq_len);
        const size_t hit_blocks = hit_tokens / engine_config.block_size;
        if (hit_blocks == 0) {
            cursor += seq_len;
            continue;
        }

        auto seq = seq_pool->get(seq_id);
        if (!seq) {
            seq = seq_pool->create(seq_id);
        }

        // If blocks are already bound for this prefix avoid duplicating refs.
        if (seq->blocks.size() >= hit_blocks) {
            cursor += seq_len;
            continue;
        }

        std::vector<size_t> token_ids_for_seq(
            batch.token_ids.begin() + cursor,
            batch.token_ids.begin() + cursor + (hit_blocks * engine_config.block_size)
        );

        std::vector<size_t> block_ids;
        ErrorCode lookup_err = prefix_cache_manager->get_cache_block_ids(token_ids_for_seq, block_ids);
        if (lookup_err != ErrorCode::SUCCESS || block_ids.size() < hit_blocks) {
            LOG_ERROR("Worker failed to lookup prefix cache block ids for sequence " + std::to_string(seq_id));
            return ErrorCode::UNKNOWN_ERROR;
        }

        for (size_t b = seq->blocks.size(); b < hit_blocks; ++b) {
            const size_t block_id = block_ids[b];
            auto cache_block = cache_manager->get_cache_block(block_id);
            if (!std::holds_alternative<std::shared_ptr<CacheBlock>>(cache_block)) {
                LOG_ERROR("Worker failed to get cache block " + std::to_string(block_id) + " for prefix cache binding.");
                return ErrorCode::UNKNOWN_ERROR;
            }

            ErrorCode add_ref_err = cache_manager->add_block_ref(block_id);
            if (add_ref_err != ErrorCode::SUCCESS) {
                LOG_ERROR("Worker failed to add cache block ref for block " + std::to_string(block_id));
                return add_ref_err;
            }
            seq->blocks.push_back(std::get<std::shared_ptr<CacheBlock>>(cache_block));
        }

        cursor += seq_len;
    }

    return ErrorCode::SUCCESS;
}

ErrorCode Worker::trim_prefill_batch_after_prefix_bind(Batch& batch) {
    if (batch.batch_size == 0) {
        batch.token_ids.clear();
        batch.token_positions.clear();
        batch.sequence_ids.clear();
        batch.num_tokens = 0;
        return ErrorCode::SUCCESS;
    }

    if (batch.prefix_hit_tokens_per_seq.size() != batch.batch_size ||
        batch.max_token_positions.size() != batch.batch_size) {
        LOG_ERROR("Worker received invalid prefix hit metadata for prefill trimming.");
        return ErrorCode::INVALID_INPUT;
    }

    std::vector<size_t> trimmed_token_ids;
    std::vector<size_t> trimmed_token_positions;
    std::vector<size_t> trimmed_sequence_ids;
    std::vector<size_t> trimmed_max_token_positions;
    std::vector<size_t> normalized_hits;

    trimmed_token_ids.reserve(batch.token_ids.size());
    trimmed_token_positions.reserve(batch.token_positions.size());
    trimmed_sequence_ids.reserve(batch.sequence_ids.size());
    trimmed_max_token_positions.reserve(batch.batch_size);
    normalized_hits.reserve(batch.batch_size);

    size_t cursor = 0;
    for (size_t seq_idx = 0; seq_idx < batch.batch_size; ++seq_idx) {
        const size_t seq_len = batch.max_token_positions[seq_idx] + 1;
        if (cursor + seq_len > batch.token_ids.size() ||
            cursor + seq_len > batch.token_positions.size() ||
            cursor + seq_len > batch.sequence_ids.size()) {
            LOG_ERROR("Worker prefill batch layout invalid while trimming bound prefix tokens.");
            return ErrorCode::INVALID_INPUT;
        }

        size_t hit_tokens = std::min(batch.prefix_hit_tokens_per_seq[seq_idx], seq_len);
        //binding is block granularity; only trim whole-block hits from compute payload.
        hit_tokens = (hit_tokens / engine_config.block_size) * engine_config.block_size;

        if (hit_tokens >= seq_len) {
            cursor += seq_len;
            continue;
        }

        const size_t keep_begin = cursor + hit_tokens;
        const size_t keep_len = seq_len - hit_tokens;

        trimmed_token_ids.insert(
            trimmed_token_ids.end(),
            batch.token_ids.begin() + keep_begin,
            batch.token_ids.begin() + keep_begin + keep_len
        );
        trimmed_token_positions.insert(
            trimmed_token_positions.end(),
            batch.token_positions.begin() + keep_begin,
            batch.token_positions.begin() + keep_begin + keep_len
        );
        trimmed_sequence_ids.insert(
            trimmed_sequence_ids.end(),
            batch.sequence_ids.begin() + keep_begin,
            batch.sequence_ids.begin() + keep_begin + keep_len
        );

        // max_token_positions is interpreted as query-count-1 for prefill read-cache metadata.
        trimmed_max_token_positions.push_back(keep_len - 1);
        normalized_hits.push_back(hit_tokens);

        cursor += seq_len;
    }

    batch.token_ids.swap(trimmed_token_ids);
    batch.token_positions.swap(trimmed_token_positions);
    batch.sequence_ids.swap(trimmed_sequence_ids);
    batch.max_token_positions.swap(trimmed_max_token_positions);
    batch.prefix_hit_tokens_per_seq.swap(normalized_hits);
    batch.batch_size = batch.max_token_positions.size();
    batch.num_tokens = batch.token_ids.size();

    return ErrorCode::SUCCESS;
}