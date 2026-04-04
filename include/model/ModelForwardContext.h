#pragma once

#include "Workspace.h"
#include "SequencePool.h"

struct ModelForwardContext {
    Workspace* workspace = nullptr;
    SequencePool* seq_pool = nullptr;

    size_t start_layer = 0;
    size_t end_layer = 0;
    void* external_hidden_in = nullptr;
    void** external_hidden_out = nullptr;
};
