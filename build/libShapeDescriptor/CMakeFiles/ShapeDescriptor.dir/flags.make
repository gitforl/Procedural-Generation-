# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# compile CUDA with /usr/local/cuda/bin/nvcc
# compile CXX with /usr/bin/g++
CUDA_DEFINES = -DDESCRIPTOR_CUDA_KERNELS_ENABLED

CUDA_INCLUDES = -I/lhome/lukashg/libShapeDescriptor/lib/nvidia-samples-common -I/usr/local/cuda/include -I/lhome/lukashg/libShapeDescriptor/src -I/lhome/lukashg/libShapeDescriptor/lib/glm -I/lhome/lukashg/libShapeDescriptor/lib/lodepng -I/lhome/lukashg/libShapeDescriptor/lib/arrrgh -I/lhome/lukashg/libShapeDescriptor/lib/json -I/lhome/lukashg/libShapeDescriptor/lib/fast-lzma2/src -I/lhome/lukashg/libShapeDescriptor/lib -I/lhome/lukashg/libShapeDescriptor/lib/fast-obj

CUDA_FLAGS = -g --generate-code=arch=compute_52,code=[compute_52,sm_52] --generate-code=arch=compute_60,code=[compute_60,sm_60] --generate-code=arch=compute_61,code=[compute_61,sm_61] --generate-code=arch=compute_70,code=[compute_70,sm_70] --generate-code=arch=compute_75,code=[compute_75,sm_75]

CXX_DEFINES = -DDESCRIPTOR_CUDA_KERNELS_ENABLED

CXX_INCLUDES = -I/lhome/lukashg/libShapeDescriptor/lib/nvidia-samples-common -I/usr/local/cuda/include -I/lhome/lukashg/libShapeDescriptor/src -I/lhome/lukashg/libShapeDescriptor/lib/glm -I/lhome/lukashg/libShapeDescriptor/lib/lodepng -I/lhome/lukashg/libShapeDescriptor/lib/arrrgh -I/lhome/lukashg/libShapeDescriptor/lib/json -I/lhome/lukashg/libShapeDescriptor/lib/fast-lzma2/src -I/lhome/lukashg/libShapeDescriptor/lib -I/lhome/lukashg/libShapeDescriptor/lib/fast-obj

CXX_FLAGS = -g
