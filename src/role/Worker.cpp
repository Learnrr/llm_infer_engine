#include "role/Worker.h"
#include "executor/PiplineExecutor.h"
#include "channel/ChannelManager.h"
#include "utils/logger.h"
#include "Sequence.h"
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
    from_scheduler = get_or_null("scheduler_to_worker_" + std::to_string(rank));
    to_scheduler = get_or_null("worker_" + std::to_string(rank) + "_to_scheduler");

    from_prev_worker = nullptr;
    to_next_worker = nullptr;
    if (rank > 0) {
        from_prev_worker = get_or_null("worker_" + std::to_string(rank - 1) + "_to_worker_" + std::to_string(rank));
    }
    if (rank + 1 < engine_config.world_size) {
        to_next_worker = get_or_null("worker_" + std::to_string(rank) + "_to_worker_" + std::to_string(rank + 1));
    }
}

void Worker::run(){
    LOG_INFO("Worker started running.");
    work();
    cleanup_retained_events();
    LOG_INFO("Worker stopped running.");
}

void Worker::request_stop() {
    stop_requested.store(true);

    // Wake the blocking receive() in work() by injecting a local STOP message.
    Channel* input = engine_config.is_first_stage() ? from_scheduler : from_prev_worker;
    if (input == nullptr) {
        LOG_ERROR("Worker cannot request stop: input channel is null.");
        return;
    }

    ForwardMessage stop_message;
    stop_message.op_type = ForwardOp::STOP;
    input->send(stop_message);
}

void Worker::work() {
    while (!stop_requested.load()) {
        ForwardMessage message;
        ErrorCode recv_error = receive(message);
        if (recv_error != ErrorCode::SUCCESS) {
            if (stop_requested.load()) {
                break;
            }
            LOG_ERROR("Worker failed to receive forward message.");
            continue;
        }
        if (message.op_type == ForwardOp::STOP) {
            ForwardMessage stop_response;
            stop_response.op_type = engine_config.is_last_stage() ? ForwardOp::DONE : ForwardOp::STOP;
            ErrorCode send_error = send(stop_response);
            if (send_error != ErrorCode::SUCCESS) {
                LOG_ERROR("Worker failed to forward STOP control response.");
            }
            stop_requested.store(true);
            break;
        }
        if(message.op_type == ForwardOp::INVALID) {
            send(message);
            continue;
        }
        if (message.op_type == ForwardOp::FREE_SEQ) {
            freeFinishedSequencesOnWorkers(message.batch.sequence_ids);
            continue;
        }

        if (message.op_type == ForwardOp::RELEASE_EVENTS_FAILED) {
            ForwardMessage failure_response;
            failure_response.batch.batch_id = message.batch.batch_id;
            if (engine_config.is_last_stage()) {
                failure_response.op_type = ForwardOp::RELEASE_EVENTS_FAILED;
            } else {
                failure_response.op_type = ForwardOp::RELEASE_EVENTS_FAILED;
            }
            ErrorCode send_error = send(failure_response);
            if (send_error != ErrorCode::SUCCESS) {
                LOG_ERROR("Worker failed to forward RELEASE_EVENTS_FAILED control response.");
            }
            continue;
        }

        if (message.op_type == ForwardOp::RELEASE_EVENTS) {
            bool release_error = false;
            if (!engine_config.is_last_stage()) {
                const size_t release_batch_id = message.batch.batch_id;
                auto it = retained_outgoing_events.find(release_batch_id);
                if (it != retained_outgoing_events.end()) {
                    cudaEvent_t event_to_release = it->second;
                    retained_outgoing_events.erase(it);
                    cudaError_t destroy_err = cudaEventDestroy(event_to_release);
                    if (destroy_err != cudaSuccess) {
                        LOG_ERROR("Worker failed to destroy retained outgoing event: " + std::string(cudaGetErrorString(destroy_err)));
                        release_error = true;
                    }
                } else {
                    LOG_DEBUG("Worker received RELEASE_EVENTS with no retained outgoing event for batch_id=" + std::to_string(release_batch_id) + "; treat as best-effort success.");
                }
            }

            ForwardMessage release_response;
            release_response.batch.batch_id = message.batch.batch_id;
            if (engine_config.is_last_stage()) {
                release_response.op_type = release_error ? ForwardOp::RELEASE_EVENTS_FAILED : ForwardOp::DONE;
            } else {
                release_response.op_type = release_error ? ForwardOp::RELEASE_EVENTS_FAILED : ForwardOp::RELEASE_EVENTS;
            }
            ErrorCode send_error = send(release_response);
            if (send_error != ErrorCode::SUCCESS) {
                LOG_ERROR("Worker failed to forward RELEASE_EVENTS control response.");
            }
            continue;
        }
           
        Batch batch = message.batch;
        batch.num_tokens = batch.num_tokens > 0 ? batch.num_tokens : batch.token_ids.size();
        if (batch.num_tokens == 0) {
            LOG_ERROR("Worker received empty batch message.");
            continue;
        }
        if (batch.sequence_ids.size() < batch.num_tokens || batch.token_positions.size() < batch.num_tokens) {
            LOG_ERROR("Worker received malformed batch: sequence_ids/token_positions size mismatch");
            continue;
        }
        
        const bool is_prefill = (message.op_type == ForwardOp::PREFILL);
        const bool is_decode = (message.op_type == ForwardOp::DECODE);
        if (!is_prefill && !is_decode) {
            ForwardMessage invalid_response;
            invalid_response.op_type = ForwardOp::INVALID;
            send(invalid_response);
            continue;
        }

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

        bool alloc_failed = false;
        for (const auto& [seq_id, required_blk_idx] : seq_required_blocks) {
            auto seq = seq_pool->get(seq_id);
            if (!seq) {
                seq = seq_pool->create(seq_id);
            }

            auto& blocks = seq->blocks;
            while (blocks.size() < required_blk_idx) {
                auto result = cache_manager->allocate_cache_block();
                if (std::holds_alternative<std::shared_ptr<CacheBlock>>(result)) {
                    blocks.push_back(std::get<std::shared_ptr<CacheBlock>>(result));
                } else {
                    LOG_ERROR("Worker failed to allocate KV block for sequence " + std::to_string(seq_id));
                    alloc_failed = true;
                    break;
                }
            }
            if (alloc_failed) {
                break;
            }
        }
        if (alloc_failed) {
            continue;
        }

        PiplineExecutor* pipeline_executor = dynamic_cast<PiplineExecutor*>(model_executor.get());
        auto run_forward = [&](void* external_hidden_in, void** external_hidden_out) -> bool {
            const bool needs_stage_io =
                engine_config.enable_pipeline_parallel &&
                (external_hidden_in != nullptr || external_hidden_out != nullptr);
            if (is_prefill) {
                if (needs_stage_io) {
                    if(pipeline_executor == nullptr) {
                        LOG_ERROR("Worker model executor is not a PiplineExecutor but received external hidden states.");
                        return false;
                    }
                    pipeline_executor->run_prefill(batch, external_hidden_in, external_hidden_out);
                } else {
                    model_executor->run_prefill(batch);
                }
            } else {
                if (needs_stage_io) {
                    if(pipeline_executor == nullptr) {
                        LOG_ERROR("Worker model executor is not a PiplineExecutor but received external hidden states.");
                        return false;
                    }
                    pipeline_executor->run_decode(batch, external_hidden_in, external_hidden_out);
                } else {
                    model_executor->run_decode(batch);
                }
            }
            return true;
        };
        void* external_hidden_in = nullptr;
        void* external_hidden_out = nullptr;
        void** external_hidden_out_ptr = engine_config.is_last_stage() ? nullptr : &external_hidden_out;

        if (message.has_cuda_ipc_handle) {
            cudaIpcMemHandle_t handle = message.cuda_ipc_handle();
            cudaEvent_t incoming_ready_event = nullptr;
            
            cudaError_t err = cudaIpcOpenMemHandle(&external_hidden_in, handle, cudaIpcMemLazyEnablePeerAccess);
            if (err != cudaSuccess) {
                LOG_ERROR("Worker failed to open CUDA IPC handle: " + std::string(cudaGetErrorString(err)));
                continue;
            }

            if (message.has_cuda_ipc_event_handle) {
                cudaIpcEventHandle_t event_handle = message.cuda_ipc_event_handle();
                cudaError_t open_event_err = cudaIpcOpenEventHandle(&incoming_ready_event, event_handle);
                if (open_event_err != cudaSuccess) {
                    LOG_ERROR("Worker failed to open CUDA IPC event handle: " + std::string(cudaGetErrorString(open_event_err)));
                    cudaIpcCloseMemHandle(external_hidden_in);
                    continue;
                }

                cudaError_t wait_err = cudaStreamWaitEvent(0, incoming_ready_event, 0);
                if (wait_err != cudaSuccess) {
                    LOG_ERROR("Worker failed to wait on CUDA IPC event: " + std::string(cudaGetErrorString(wait_err)));
                    cudaEventDestroy(incoming_ready_event);
                    cudaIpcCloseMemHandle(external_hidden_in);
                    continue;
                }
            }

            const bool ok = run_forward(external_hidden_in, external_hidden_out_ptr);

            if (incoming_ready_event != nullptr) {
                cudaError_t destroy_event_err = cudaEventDestroy(incoming_ready_event);
                if (destroy_event_err != cudaSuccess) {
                    LOG_ERROR("Worker failed to destroy imported IPC event: " + std::string(cudaGetErrorString(destroy_event_err)));
                }
            }

            cudaError_t close_err = cudaIpcCloseMemHandle(external_hidden_in);
            if (close_err != cudaSuccess) {
                LOG_ERROR("Worker failed to close CUDA IPC handle: " + std::string(cudaGetErrorString(close_err)));
            }
            if (!ok) {
                continue;
            }
        } else if (!run_forward(nullptr, external_hidden_out_ptr)) {
            continue;
        }

        //build message for scheduler
        ForwardMessage response;
        if(engine_config.is_last_stage()){
            response.op_type = ForwardOp::DONE;
            response.batch = batch;
            // for the last stage worker, already append sampled token ids in model execution
            response.batch.token_ids = batch.token_ids;
            response.batch.sampled_token_ids = batch.sampled_token_ids;
            response.batch.num_tokens = response.batch.token_ids.size();
        } else { //build message for next stage worker
            response.op_type = is_prefill ? ForwardOp::PREFILL : ForwardOp::DECODE;
            response.batch = batch;

            if (external_hidden_out == nullptr) {
                LOG_ERROR("Worker did not produce external_hidden_out for next pipeline stage.");
                continue;
            }
            // need to set the CUDA IPC handle for the output hidden states for non-last stage workers
            cudaIpcMemHandle_t out_handle{};
            cudaError_t ipc_err = cudaIpcGetMemHandle(&out_handle, external_hidden_out);
            if (ipc_err != cudaSuccess) {
                LOG_ERROR("Worker failed to get CUDA IPC handle for external_hidden_out: " + std::string(cudaGetErrorString(ipc_err)));
                continue;
            }
            // need to create an IPC event to signal the next stage worker when the hidden states are ready
            cudaEvent_t outgoing_ready_event = nullptr;
            cudaError_t create_err = cudaEventCreateWithFlags(
                &outgoing_ready_event,
                cudaEventDisableTiming | cudaEventInterprocess
            );
            if (create_err != cudaSuccess) {
                LOG_ERROR("Worker failed to create IPC event for outgoing hidden states: " + std::string(cudaGetErrorString(create_err)));
                continue;
            }

            cudaIpcEventHandle_t outgoing_ready_event_handle{};
            cudaError_t export_err = cudaIpcGetEventHandle(&outgoing_ready_event_handle, outgoing_ready_event);
            if (export_err != cudaSuccess) {
                LOG_ERROR("Worker failed to export IPC event handle: " + std::string(cudaGetErrorString(export_err)));
                cudaEventDestroy(outgoing_ready_event);
                continue;
            }

            cudaError_t record_err = cudaEventRecord(outgoing_ready_event, 0);
            if (record_err != cudaSuccess) {
                LOG_ERROR("Worker failed to record outgoing IPC event: " + std::string(cudaGetErrorString(record_err)));
                cudaEventDestroy(outgoing_ready_event);
                continue;
            }

            response.set_cuda_ipc_handle(out_handle, &outgoing_ready_event_handle);

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


void Worker::freeFinishedSequencesOnWorkers(const std::vector<size_t>& sequence_ids) {
    for (size_t seq_id : sequence_ids) {
        auto seq = seq_pool->get(seq_id);
        if (!seq) {
            continue;
        }
        for (const auto& block : seq->blocks) {
            if (block) {
                cache_manager->free_cache_block(block->block_id);
            }
        }
        seq->blocks.clear();
        seq_pool->erase(seq_id);
    }

    ForwardMessage cleanup_response;
    cleanup_response.batch.sequence_ids = sequence_ids;
    if (engine_config.is_last_stage()) {
        cleanup_response.op_type = ForwardOp::DONE;
    } else {
        cleanup_response.op_type = ForwardOp::FREE_SEQ;
    }
    ErrorCode send_error = send(cleanup_response);
    if (send_error != ErrorCode::SUCCESS) {
        LOG_ERROR("Worker failed to forward FREE_SEQ cleanup response.");
    }    
}