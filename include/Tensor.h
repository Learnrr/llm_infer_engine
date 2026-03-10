#include <vector>


enum class DataType {
    FLOAT32,
    FLOAT16
};

class Tensor{
    public:
        void* data;
        size_t size;
        DataType dtype;
        vector<size_t> shape;
        string device;

        Tensor(size_t size, void* data_ptr, vector<size_t> shape, DataType dtype, string device = "cpu") 
            : size(size), shape(shape), dtype(dtype), data(data_ptr), device(device) {}

        ~Tensor() {}

        Tensor(const Tensor& other) : size(other.size), shape(other.shape), device(other.device) {
            data = new float[size];
            std::copy(other.data, other.data + size, data);
        }

        void view(vector<size_t> new_shape) {
            if(data != nullptr){
                shape = new_shape;
            }    
        }
        Tensor transpose() {
            Tensor out(size,nullptr,{shape[1],shape[0]},dtype,device);
            out.data = new char[size];
            if(data != nullptr){
                if(shape.size() == 2){
                    for(int i = 0;i < shape[0]; ++i){
                        for(int j = 0;j < shape[1]; ++j){
                            *(out.data + j * shape[0] + i) = *(data + i * shape[1] + j);
                        }
                    }
                }
            }
            return out;
        }
        bool& operator==(const Tensor& other) {
            if (size != other.size || shape != other.shape) {
                return false;
            } else {
                for (size_t i = 0; i < size; ++i) {
                    if (data[i] != other.data[i]) {
                        return false;
                    }
                }
                return true;
            }
            return *this;
        }

}