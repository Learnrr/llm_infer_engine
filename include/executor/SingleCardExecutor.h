#pragma once

#include "executor/Executor.h"
#include "model/IModel.h"
#include "SequencePool.h"
#include "Workspace.h"

class SingleCardExecutor : public Executor {
public:
    SingleCardExecutor(
        IModel* model, 
        Workspace* workspace, 
        SequencePool* seq_pool = nullptr
    )
        : model(model), 
        workspace(workspace), 
        seq_pool(seq_pool) {}

    void run_prefill(Batch& batch) override{
        run_prefill(batch, workspace);
    }
    void run_decode(Batch& batch) override{
        run_decode(batch, workspace);
    }

    void run_prefill(Batch& batch, Workspace* workspace);
    void run_decode(Batch& batch, Workspace* workspace);

private:
    IModel* model;
    Workspace* workspace;
    SequencePool* seq_pool;
};