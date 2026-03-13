#include "Batch.h"
#include "Workspace.h"
#include "ModelConfig.h"
class IModel{
    public:
        virtual void init(ModelConfig config) = 0;
        virtual void prefill_forward(Batch& batch, Workspace& workspace) = 0;
        virtual void decode_forward(Batch& batch, Workspace& workspace) = 0;
        virtual void load_weights(const char* model_path) = 0;
        virtual ~IModel() {}
};

class ModelFactory {
    public:
        static std::unique_ptr<IModel> create_model(const std::string& model_name) {
            if (model_name == "QWEN") {
                return std::make_unique<QWEN_Model>();
            }
            // Add more models here as needed
            return nullptr; // Return nullptr if model name is not recognized
        }
};