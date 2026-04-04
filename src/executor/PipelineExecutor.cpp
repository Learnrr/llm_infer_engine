#include "executor/PiplineExecutor.h"
#include "model/ModelForwardContext.h"

void PiplineExecutor::run_prefill(Batch& batch) {
    run_prefill(batch, nullptr);
}

void PiplineExecutor::run_decode(Batch& batch) {
    run_decode(batch, nullptr);
}


void PiplineExecutor::run_prefill(Batch& batch, void* external_hidden_in, void** external_hidden_out) {
    ModelForwardContext context;
    context.workspace = workspace;
    context.seq_pool = seq_pool;
    context.start_layer = stage_start_layer;
    context.end_layer = stage_end_layer;
    context.external_hidden_in = external_hidden_in;
    context.external_hidden_out = external_hidden_out;
    model->stage_prefill_forward(
        batch,
        context
    );
}

void PiplineExecutor::run_decode(Batch& batch, void* external_hidden_in, void** external_hidden_out) {
    ModelForwardContext context;
    context.workspace = workspace;
    context.seq_pool = seq_pool;
    context.start_layer = stage_start_layer;
    context.end_layer = stage_end_layer;
    context.external_hidden_in = external_hidden_in;
    context.external_hidden_out = external_hidden_out;
    model->stage_decode_forward(
        batch,
        context
    );
}