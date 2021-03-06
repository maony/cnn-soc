/*
    cnn setting



*/
//#define CONV
//#define PRELU
//#define MAX_POOLING
//#define INNER_PRODUCT
#define SIGMOID

//#define RELU
//#define SOFTMAX
//#define BIAS

#ifdef CONV
#define EXT_IM2COL
#define BS_IM2COL 1
#include "im2col.cl"

#define GROUP_XX
#define BS_GEMM 1
#define SIMD_GEMM 1
#include "gemm.cl"
#endif

#ifdef PRELU
#define BS_PRELU 1
#include "prelu.cl"
#endif

#ifdef MAX_POOLING
#define BS_MAX_POOLING 1
#include "max_pooling.cl"
#endif

#ifdef INNER_PRODUCT
#define BS_INNER_PRODUCT 1
#include "inner_product.cl"
#endif

#ifdef SIGMOID
#define BS_SIGMOID 1
#include "sigmoid.cl"
#endif

#ifdef RELU
#define BS_RELU 1
#include "relu.cl"
#endif

#ifdef SOFTMAX
#define BS_SOFTMAX 1
#include "softmax.cl"
#endif

#ifdef BIAS
#define BS_BIAS 1
#include "bias.cl"
#endif
