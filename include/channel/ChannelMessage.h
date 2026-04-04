#pragma once
#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <cuda_runtime_api.h>
#include "Batch.h"

class ChannelMessage {
    public:
        virtual ~ChannelMessage() = default;
        virtual std::vector<char> serialize() const = 0;
        virtual void deserialize(const std::vector<char>& data) = 0;
};

enum class ForwardOp : uint8_t {
    INVALID = 0,
    PREFILL = 1,
    DECODE = 2,
    STOP = 3,
    DONE = 4,
    FREE_SEQ = 5,
    RELEASE_EVENTS = 6,
    RELEASE_EVENTS_FAILED = 7
};

struct ForwardMessage : public ChannelMessage {
    ForwardOp op_type = ForwardOp::INVALID;
    Batch batch;

    bool has_cuda_ipc_handle = false;
    bool has_cuda_ipc_event_handle = false;
    std::array<char, sizeof(cudaIpcMemHandle_t)> cuda_ipc_handle_bytes{};
    std::array<char, sizeof(cudaIpcEventHandle_t)> cuda_ipc_event_handle_bytes{};

    void set_cuda_ipc_handle(const cudaIpcMemHandle_t& handle, const cudaIpcEventHandle_t* event_handle = nullptr) {
        std::memcpy(cuda_ipc_handle_bytes.data(), &handle, sizeof(handle));
        has_cuda_ipc_handle = true;

        if (event_handle) {
            std::memcpy(cuda_ipc_event_handle_bytes.data(), event_handle, sizeof(*event_handle));
            has_cuda_ipc_event_handle = true;
        } else {
            has_cuda_ipc_event_handle = false;
            cuda_ipc_event_handle_bytes.fill(0);
        }
    }

    void set_cuda_ipc_event_handle(const cudaIpcEventHandle_t& event_handle) {
        std::memcpy(cuda_ipc_event_handle_bytes.data(), &event_handle, sizeof(event_handle));
        has_cuda_ipc_event_handle = true;
    }

    cudaIpcMemHandle_t cuda_ipc_handle() const {
        cudaIpcMemHandle_t handle{};
        std::memcpy(&handle, cuda_ipc_handle_bytes.data(), sizeof(handle));
        return handle;
    }

    cudaIpcEventHandle_t cuda_ipc_event_handle() const {
        cudaIpcEventHandle_t handle{};
        std::memcpy(&handle, cuda_ipc_event_handle_bytes.data(), sizeof(handle));
        return handle;
    }

    std::vector<char> serialize() const override {
        auto vec_bytes = [](const std::vector<size_t>& v) {
            return sizeof(size_t) + v.size() * sizeof(size_t);
        };

        const uint8_t has_handle = has_cuda_ipc_handle ? 1 : 0;
        const uint8_t has_event_handle = has_cuda_ipc_event_handle ? 1 : 0;

        const size_t total =
            sizeof(op_type) +
            sizeof(batch.batch_id) +
            vec_bytes(batch.token_ids) +
            vec_bytes(batch.token_positions) +
            vec_bytes(batch.sampled_token_ids) +
            vec_bytes(batch.sequence_ids) +
            vec_bytes(batch.max_token_positions) +
            sizeof(batch.num_tokens) +
            sizeof(batch.batch_size) +
            sizeof(has_handle) +
            cuda_ipc_handle_bytes.size() +
            sizeof(has_event_handle) +
            cuda_ipc_event_handle_bytes.size();

        std::vector<char> data(total);

        auto write_scalar = [&data](size_t& offset, auto value) {
            std::memcpy(data.data() + offset, &value, sizeof(value));
            offset += sizeof(value);
        };

        auto write_vector = [&data, &write_scalar](size_t& offset, const std::vector<size_t>& v) {
            const size_t len = v.size();
            write_scalar(offset, len);
            if (len > 0) {
                std::memcpy(data.data() + offset, v.data(), len * sizeof(size_t));
                offset += len * sizeof(size_t);
            }
        };

        size_t offset = 0;
        write_scalar(offset, op_type);
        write_scalar(offset, batch.batch_id);

        write_vector(offset, batch.token_ids);
        write_vector(offset, batch.token_positions);
        write_vector(offset, batch.sampled_token_ids);
        write_vector(offset, batch.sequence_ids);
        write_vector(offset, batch.max_token_positions);
        write_scalar(offset, batch.num_tokens);
        write_scalar(offset, batch.batch_size);
        write_scalar(offset, has_handle);
        write_scalar(offset, has_event_handle);

        std::memcpy(
            data.data() + offset,
            cuda_ipc_handle_bytes.data(),
            cuda_ipc_handle_bytes.size()
        );
        offset += cuda_ipc_handle_bytes.size();

        if (has_cuda_ipc_event_handle) {
            std::memcpy(
                data.data() + offset,
                cuda_ipc_event_handle_bytes.data(),
                cuda_ipc_event_handle_bytes.size()
            );
            offset += cuda_ipc_event_handle_bytes.size();
        }

        return data;
    }

    void deserialize(const std::vector<char>& data) override {
        op_type = ForwardOp::INVALID;
        batch.token_ids.clear();
        batch.token_positions.clear();
        batch.sampled_token_ids.clear();
        batch.sequence_ids.clear();
        batch.max_token_positions.clear();
        batch.num_tokens = 0;
        batch.batch_size = 0;
        batch.batch_id = 0;

        has_cuda_ipc_handle = false;
        cuda_ipc_handle_bytes.fill(0);
        has_cuda_ipc_event_handle = false;
        cuda_ipc_event_handle_bytes.fill(0);


        auto read_scalar = [&data](size_t& offset, auto& out) -> bool {
            if (offset + sizeof(out) > data.size()) {
                return false;
            }
            std::memcpy(&out, data.data() + offset, sizeof(out));
            offset += sizeof(out);
            return true;
        };

        auto read_vector = [&data, &read_scalar](size_t& offset, std::vector<size_t>& out) -> bool {
            size_t len = 0;
            if (!read_scalar(offset, len)) {
                return false;
            }
            if (offset + len * sizeof(size_t) > data.size()) {
                return false;
            }
            out.resize(len);
            if (len > 0) {
                std::memcpy(out.data(), data.data() + offset, len * sizeof(size_t));
                offset += len * sizeof(size_t);
            }
            return true;
        };

        size_t offset = 0;
        if (!read_scalar(offset, op_type)) {
            return;
        }

        if (!read_scalar(offset, batch.batch_id)) {
            return;
        }

        if (!read_vector(offset, batch.token_ids)) {
            return;
        }
        if (!read_vector(offset, batch.token_positions)) {
            return;
        }
        if (!read_vector(offset, batch.sampled_token_ids)) {
            return;
        }
        if (!read_vector(offset, batch.sequence_ids)) {
            return;
        }
        if (!read_vector(offset, batch.max_token_positions)) {
            return;
        }
        if (!read_scalar(offset, batch.num_tokens)) {
            return;
        }
        if (!read_scalar(offset, batch.batch_size)) {
            return;
        }
        uint8_t has_handle = 0;
        if (!read_scalar(offset, has_handle)) {
            return;
        }
        has_cuda_ipc_handle = (has_handle != 0);

        uint8_t has_event_handle = 0;
        if (!read_scalar(offset, has_event_handle)) {
            return;
        }
        has_cuda_ipc_event_handle = (has_event_handle != 0);

        if (offset + cuda_ipc_handle_bytes.size() > data.size()) {
            return;
        }
        std::memcpy(
            cuda_ipc_handle_bytes.data(),
            data.data() + offset,
            cuda_ipc_handle_bytes.size()
        );
        offset += cuda_ipc_handle_bytes.size();

        if (has_cuda_ipc_event_handle) {
            if (offset + cuda_ipc_event_handle_bytes.size() > data.size()) {
                return;
            }
            std::memcpy(
                cuda_ipc_event_handle_bytes.data(),
                data.data() + offset,
                cuda_ipc_event_handle_bytes.size()
            );
            offset += cuda_ipc_event_handle_bytes.size();
        }
    }
};

using TensorMetadata = ForwardMessage;
using ControlMessage = ForwardMessage;