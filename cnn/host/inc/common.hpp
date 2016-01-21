#ifndef __COMMON_H__
#define __COMMON_H__

#define IM2COL
#define GEMM
#define PRELU
#define MAX_POOLING
//#define RELU
//#define INNER_PRODUCT
//#define SOFTMAX
//#define BIAS

// The set of simultaneous kernels
enum KERNELS {
#ifdef IM2COL
    K_IM2COL,
#endif
#ifdef GEMM
    K_GEMM,
#endif
#ifdef PRELU
    K_PRELU,
#endif
#ifdef MAX_POOLING
    K_MAX_POOLING,
#endif
#ifdef INNER_PRODUCT
    K_INNER_PRODUCT,
#endif
#ifdef RELU
    K_RELU,
#endif
#ifdef BIAS
    K_BIAS,
#endif
#ifdef SOFTMAX
    K_SOFTMAX,
#endif
    K_NUM_KERNELS
};

using namespace aocl_utils;

static const char* kernel_names[K_NUM_KERNELS+1] =
{
#ifdef IM2COL
    "im2col",
#endif
#ifdef GEMM
    "gemm",
#endif
#ifdef PRELU
    "prelu",
#endif
#ifdef MAX_POOLING
    "max_pooling",
#endif
#ifdef INNER_PRODUCT
    "inner_product",
#endif
#ifdef RELU
    "relu",
#endif
#ifdef BIAS
    "bias",
#endif
#ifdef SOFTMAX
    "softmax",
#endif
    "null"
};

#endif

