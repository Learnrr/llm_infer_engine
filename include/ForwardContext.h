#pragma once

#include "Batch.h"
#include "Workspace.h"
#include "ModelConfig.h"
struct ForwardContext {
    size_t layer_id;
    Batch* batch;
    Workspace* workspace;
    ModelConfig* config;


};