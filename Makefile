
ROOT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
TARGET := $(ROOT_DIR)/cpp_engine.so
SRC_DIR := $(ROOT_DIR)/src
CU_DIR := $(ROOT_DIR)/kernel
INC_DIR := $(ROOT_DIR)/include
BUILD_DIR := $(ROOT_DIR)/build
CUDA_HOME ?= /usr/local/cuda
CUDA_INC := $(CUDA_HOME)/include
CUDA_LIB := $(CUDA_HOME)/lib64
CPPFLAGS := -Wall -fPIC -I$(INC_DIR) -I$(INC_DIR)/layer -I$(INC_DIR)/utils -I$(INC_DIR)/model -I$(INC_DIR)/kernel \
	-I$(CUDA_INC) \
	$(shell python3 -m pybind11 --includes)
NVCCFLAGS := -O2 -Xcompiler -fPIC -I$(INC_DIR) -I$(INC_DIR)/kernel -I$(CUDA_INC)
LDFLAGS := -shared
PYTHON := python3
LDLIBS := $(shell $(PYTHON)-config --ldflags) -L$(CUDA_LIB) -lcudart
CXX := g++
NVCC := nvcc

SRCS_CPP = $(shell find $(SRC_DIR) -name "*.cpp")
SRCS_CU = $(shell find $(CU_DIR) -name "*.cu")
OBJS_CPP = $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRCS_CPP))
OBJS_CU = $(patsubst $(CU_DIR)/%.cu,$(BUILD_DIR)/%.o,$(SRCS_CU))


$(TARGET): $(OBJS_CPP) $(OBJS_CU)
	$(CXX) $(LDFLAGS) -o $@ $(OBJS_CPP) $(OBJS_CU) $(LDLIBS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(CU_DIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

all: $(TARGET)

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR) $(TARGET)
	rm -rf $(ROOT_DIR)/cpp_engine.so