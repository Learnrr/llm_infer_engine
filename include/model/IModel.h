#pragma once

#include "model/ModelForwardContext.h"
#include "llm_engine_config.h"
#include <memory>
#include <string>

class IModel{
    public:
    virtual void init(LLMEngineConfig& config) = 0;
        virtual void prefill_forward(Batch& batch, ModelForwardContext& context) = 0;
        virtual void decode_forward(Batch& batch, ModelForwardContext& context) = 0;
        virtual void load_weights(const char* model_path) = 0;
        virtual ~IModel() {}

        virtual void stage_prefill_forward(Batch& batch, ModelForwardContext& context) = 0;
        virtual void stage_decode_forward(Batch& batch, ModelForwardContext& context) = 0;

        void prefill_forward(Batch& batch, Workspace& workspace, SequencePool* seq_pool = nullptr) {
            ModelForwardContext context;
            context.workspace = &workspace;
            context.seq_pool = seq_pool;
            prefill_forward(batch, context);
        }

        void decode_forward(Batch& batch, Workspace& workspace, SequencePool* seq_pool = nullptr) {
            ModelForwardContext context;
            context.workspace = &workspace;
            context.seq_pool = seq_pool;
            decode_forward(batch, context);
        }

        void stage_prefill_forward(
            Batch& batch,
            Workspace& workspace,
            size_t start_layer,
            size_t end_layer,
            void* external_hidden_in = nullptr,
            SequencePool* seq_pool = nullptr
        ) {
            ModelForwardContext context;
            context.workspace = &workspace;
            context.seq_pool = seq_pool;
            context.start_layer = start_layer;
            context.end_layer = end_layer;
            context.external_hidden_in = external_hidden_in;
            stage_prefill_forward(batch, context);
        }

        void stage_decode_forward(
            Batch& batch,
            Workspace& workspace,
            size_t start_layer,
            size_t end_layer,
            void* external_hidden_in = nullptr,
            SequencePool* seq_pool = nullptr
        ) {
            ModelForwardContext context;
            context.workspace = &workspace;
            context.seq_pool = seq_pool;
            context.start_layer = start_layer;
            context.end_layer = end_layer;
            context.external_hidden_in = external_hidden_in;
            stage_decode_forward(batch, context);
        }
};

class ModelFactory {
    public:
        static std::unique_ptr<IModel> create_model(const std::string& model_name);
};