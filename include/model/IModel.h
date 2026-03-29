#pragma once

#include "Batch.h"
#include "Workspace.h"
#include "llm_engine_config.h"
#include <memory>
#include <string>

class IModel{
    public:
    virtual void init(LLMEngineConfig& config) = 0;
        virtual void prefill_forward(Batch& batch, Workspace& workspace) = 0;
        virtual void decode_forward(Batch& batch, Workspace& workspace) = 0;
        virtual void load_weights(const char* model_path) = 0;
        virtual ~IModel() {}
};

class ModelFactory {
    public:
        static std::unique_ptr<IModel> create_model(const std::string& model_name);
};