#ifndef __CNN_REF__
#define __CNN_REF__

#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"

#define MAX(a, b) (a < b ? b : a)
#define MIN(a, b) (a < b ? a : b)

#ifdef __ARCH_DSP__
#define NO_ALIAS    restrict
#define ASSERT(x)   _nassert(x)
#else
#define NO_ALIAS
#define ASSERT(x)   
#endif

using namespace aocl_utils;
typedef char	uchar;
//typedef int		uint;
typedef float	Dtype;

typedef struct blob_shape
{
	int num;
	int channels;
	int height;
	int width;
	float *data;
   
    blob_shape() {};
    blob_shape(int n, int c, int h, int w) {
        num = n;
        channels = c;
        height = h;
        width = w;
    };
}blob_shape_t;

typedef struct conv_param
{
	int pad_w;
	int pad_h;
	int stride_w;
	int stride_h;
    conv_param() {};
    conv_param(int ph, int pw, int sh, int sw) {
        pad_w = pw;
        pad_h = ph;
        stride_w = sw;
        stride_h = sh;
    };
}conv_param_t;

void im2col_ref(const float* data_im, float* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w); 

void load_model_param(FILE *fp, uint offset, uint num, float *param);
void conv_layer(const blob_shape_t src, const blob_shape_t filter, blob_shape_t dst, const conv_param_t param);
void conv_bias(float * NO_ALIAS data, uint row, uint col, float * NO_ALIAS bias);
void pooling_max_layer(const blob_shape_t src, const conv_param_t param, const int kernel_h, const int kernel_w, blob_shape_t dst);
void inner_product_layer(float * NO_ALIAS in_data, float * NO_ALIAS in_weight, float * NO_ALIAS in_bias, float * NO_ALIAS out_data, uint row, uint col);
void relu_layer(float * NO_ALIAS in_data, float * NO_ALIAS out_data, uint num);
void softmax_layer(float * NO_ALIAS in_data, float * NO_ALIAS out_data, uint num);
void ref_lenet(void);
size_t size_blob(blob_shape_t data);

#endif
