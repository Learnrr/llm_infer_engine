#include <string>
class ModelConfig {
    public:
        ModelConfig() {
            max_seq_len = 512;
            hidden_size = 768;
            num_attention_heads = 12;
            num_hidden_layers = 12;
            vocab_size = 30522;
        }
        int max_seq_len;
        int hidden_size;
        int num_attention_heads;
        int num_hidden_layers;
        int vocab_size;

        string model_path;

        void build_from_file(char* config_path) {
            
        }
};