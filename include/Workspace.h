#pragma once
#include "define.h"
#include "cuda_runtime.h"

struct TransformerLayerWorkspace {
    void* base;
    size_t qkv_offset;
    size_t attn_out_offset;
    size_t context_offset;
    size_t mlp_offset;

};

struct WorkspaceLayout {

    size_t hidden_offset;
    size_t hidden2_offset;

    TransformerLayerWorkspace layer_workspace;
    size_t temp_offset;

    size_t logits_offset;

    size_t total_size;
};
size_t align_size(size_t size, size_t alignment = 256) {
    return (size + alignment - 1) & ~(alignment - 1);
}
class Workspace {
    public:
        Workspace(){}

        void init(){
            size_t hidden_size = MAX_SEQ_LEN * HIDDEN_SIZE * DTYPE;
            size_t qkv_size = MAX_SEQ_LEN * 3 * HIDDEN_SIZE * DTYPE;
            size_t attn_out_size = MAX_SEQ_LEN * HIDDEN_SIZE * DTYPE;
            size_t context_size = MAX_SEQ_LEN * HIDDEN_SIZE * DTYPE;
            size_t mlp_size = MAX_SEQ_LEN * INTERMEDIATE_SIZE * DTYPE;
            size_t logits_size = MAX_SEQ_LEN * VOCAB_SIZE * DTYPE;

            size_t offset = 0;
            layout.hidden_offset = offset;
            offset += align_size(hidden_size, 256);
            layout.hidden2_offset = offset;
            offset += align_size(hidden_size, 256);
            layout.layer_workspace.qkv_offset = offset;
            offset += align_size(qkv_size, 256);
            layout.layer_workspace.attn_out_offset = offset;
            offset += align_size(attn_out_size, 256);
            layout.layer_workspace.context_offset = offset;
            offset += align_size(context_size, 256);
            layout.layer_workspace.mlp_offset = offset;
            offset += align_size(mlp_size, 256);
            layout.logits_offset = offset;
            offset += align_size(logits_size, 256);
            layout.total_size = offset;

            cudaMalloc(&workspace, layout.total_size);

            layout.layer_workspace.base = workspace;
        }

        void free() {
            cudaFree(workspace);
        }

        ~Workspace(){}

        void* get_workspace() {
            return workspace;
        }
        void* get_embedding_workspace() {
            return (void*)((char*)workspace + layout.hidden_offset);
        }
        void* get_qkv_workspace() {
            return (void*)((char*)workspace + layout.layer_workspace.qkv_offset);
        }

        void* get_logits_workspace() {
            return (void*)((char*)workspace + layout.logits_offset);
        }
        TransformerLayerWorkspace get_transformer_layer_workspace() {
            return layout.layer_workspace;
        }
        WorkspaceLayout get_layout() {
            return layout;
        }

    private:
        void* workspace;
        WorkspaceLayout layout;


};