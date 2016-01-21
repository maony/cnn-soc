// Copyright (C) 2013-2015 Altera Corporation, San Jose, California, USA. All rights reserved.
// Permission is hereby granted, free of charge, to any person obtaining a copy of this
// software and associated documentation files (the "Software"), to deal in the Software
// without restriction, including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to
// whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
// 
// This agreement shall be governed in all respects by the laws of the State of California and
// by the laws of the United States of America.

///////////////////////////////////////////////////////////////////////////////////
// This OpenCL example implements a channelizer design on an Altera FPGA.  The
// set of kernels accept data from an input channel, stream it through
// a polyphase filter bank (to reduce spectral leakage) and a 4k-point 1D FFT
// transform on an Altera FPGA.
//
// The kernels are defined in the files under the 'device' directory.  The Altera 
// Offline Compiler tool ('aoc') compiles the kernel source into a 'channelizer.aocx' 
// file containing a hardware programming image for the FPGA.  The host program 
// provides the contents of the .aocx file to the clCreateProgramWithBinary OpenCL
// API for runtime programming of the FPGA.
//
// When compiling this application, ensure that the Altera SDK for OpenCL
// is properly installed.
///////////////////////////////////////////////////////////////////////////////////

#include <assert.h>
#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <cstring>
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#include "ref_cnn.hpp"
#include "layer.hpp"
#include "common.hpp"

// ACL runtime configuration
cl_platform_id platform = NULL;
cl_device_id device = NULL;
cl_context context = NULL;
cl_command_queue queues[K_NUM_KERNELS];
cl_kernel kernels[K_NUM_KERNELS];
cl_program program = NULL;
cl_int status = 0;

#define WRITE_TXT(dat, name, c) \
    do {                            \
        FILE *fp_file;              \
        fp_file = fopen(name, "w"); \
        for(int i = 0; i < c; i++) {    \
            fprintf(fp_file, "%6.2f\n", dat[i]); \
        }   \
        fclose(fp_file);    \
    } while(0)

#define WRITE_BIN(dat, name, c) \
    do {                            \
        FILE *fp_bin;              \
        fp_bin = fopen(name, "wb"); \
        fwrite(dat, sizeof(float), c, fp_bin); \
        fclose(fp_bin);    \
    } while(0)

// Function prototypes
bool init();
void cleanup();
bool approximatelyEqual(float a, float b, float epsilon);
bool im2col_test();
void lenet_test(int sel);
void attribute_test(void);
void ocl_attribute(void);

#ifdef MAX_POOLING
void ocl_pooling_max_layer(const blob_shape_t src, const conv_param_t param, const int kernel_h, const int kernel_w, blob_shape_t dst);
#endif
void (*pfunc_pooling)(const blob_shape_t src, const conv_param_t param, const int kernel_h, const int kernel_w, blob_shape_t dst);

#if defined(IM2COL) && defined(GEMM)
void ocl_conv_layer(const blob_shape_t src, const blob_shape_t filter, blob_shape_t dst, const conv_param_t param);
#endif
void (*pfunc_conv)(const blob_shape_t src, const blob_shape_t filter, blob_shape_t dst, const conv_param_t param);

#ifdef INNER_PRODUCT
void ocl_inner_product_layer(float * NO_ALIAS in_data, float * NO_ALIAS in_weight, float * NO_ALIAS in_bias, float * NO_ALIAS out_data, uint row, uint col);
#endif
void (*pfunc_inner_product)(float * NO_ALIAS in_data, float * NO_ALIAS in_weight, float * NO_ALIAS in_bias, float * NO_ALIAS out_data, uint row, uint col);

#ifdef RELU
void ocl_relu_layer(float * NO_ALIAS in_data, float * NO_ALIAS out_data, uint num);
#endif
void (*pfunc_relu)(float * NO_ALIAS in_data, float * NO_ALIAS out_data, uint num);

#ifdef BIAS
void ocl_conv_bias(float * NO_ALIAS data, uint row, uint col, float * NO_ALIAS bias);
#endif
void (*pfunc_bias)(float * NO_ALIAS data, uint row, uint col, float * NO_ALIAS bias);

#ifdef SOFTMAX
void ocl_softmax_layer(float * NO_ALIAS in_data, float * NO_ALIAS out_data, uint num);
#endif
void (*pfunc_softmax)(float * NO_ALIAS in_data, float * NO_ALIAS out_data, uint num);

// Entry point.
int main(int argc, char **argv) {
    Options options(argc, argv);

    // Flush stdout immediately
    setbuf(stdout, NULL);

    // Setup the context, create the device and kernels...
    if(!init()) {
        return false;
    }
    printf("-----------Init complete!-------------\n");
    printf("--------------------------------------\n");
   
    pfunc_pooling       = pooling_max_layer;
    pfunc_conv          = conv_layer;
    pfunc_inner_product = inner_product_layer;
    pfunc_relu          = relu_layer;
    pfunc_bias          = conv_bias;
    pfunc_softmax       = softmax_layer;

#if defined(IM2COL) && defined(GEMM)
    pfunc_conv          = ocl_conv_layer;
#endif
#ifdef MAX_POOLING
    pfunc_pooling       = ocl_pooling_max_layer;
#endif
#ifdef INNER_PRODUCT
    pfunc_inner_product = ocl_inner_product_layer;
#endif
#ifdef RELU
    pfunc_relu          = ocl_relu_layer;
#endif
#ifdef BIAS
    pfunc_bias          = ocl_conv_bias;
#endif
#ifdef SOFTMAX
    pfunc_softmax       = ocl_softmax_layer;
#endif

    //im2col_test();
    //ref_lenet();
    //lenet_test(5);
    //attribute_test();
    ocl_attribute();

    printf("--------------------------------------\n");
    printf("-----------test complete!-------------\n");

    // Free the resources allocated
    cleanup();
    return 0;
}

void ocl_attribute(void) {
    int offset = 0;
    FILE        *fp_data, *fp_model;
    fp_data     = fopen("conv01-in.bin", "rb");
    fp_model    = fopen("attribute-model.bin", "rb");
    
#define MUL_4(a, b, c, d) a * b * c * d
    int dn = 1, dc = 3, dh = 256, dw = 256;
    int fn = 16, fc = 3, fh = 7, fw = 7;
    int ph = fh / 2, pw = fw / 2;
    int sh = 2, sw = 2;

    float *conv01_data = (float *)alignedMalloc(sizeof(float) * MUL_4(dn, dc, dh, dw));
	fread(conv01_data, sizeof(float), MUL_4(dn, dc, dh, dw), fp_data);
    cl_mem d_conv01_data = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * MUL_4(dn, dc, dh, dw), NULL, &status);
    checkError(status, "Failed to allocate input device buffer\n");
    status = clEnqueueWriteBuffer(queues[K_IM2COL], d_conv01_data, CL_TRUE, 0, sizeof(float) * MUL_4(dn, dc, dh, dw), conv01_data, 0, NULL, NULL);
    checkError(status, "Failed to copy data to device");
    clFinish(queues[K_IM2COL]);
   
#define FREE(x) \
    do {        \
        if(x) { alignedFree(x); x = NULL; } \
    } while(0)

#define INIT_CONV() \
    do {    \
        conv01_filter = (float *)alignedMalloc(sizeof(float) * MUL_4(fn, fc, fh, fw));\
        conv01_bias   = (float *)alignedMalloc(sizeof(float) * fn);\
        load_model_param(fp_model, offset, MUL_4(fn, fc, fh, fw), conv01_filter);\
	    offset += MUL_4(fn, fc, fh, fw);\
	    load_model_param(fp_model, offset, fn, conv01_bias);\
	    offset += fn;\
    } while(0)

#define INIT_PRELU() \
    do {    \
        conv01_bias   = (float *)alignedMalloc(sizeof(float) * dc);\
        load_model_param(fp_model, offset, dc, conv01_bias);\
	    offset += dc;\
    } while(0)

    float *conv01_filter;
    float *conv01_bias;
    // -----------init start------------------
    // conv01
    INIT_CONV();
    ConvLayer conv01(dn, dc, dh, dw, ph, pw, sh, sw);
    conv01.init_weight(fn, fh, fw, conv01_filter, conv01_bias);
    FREE(conv01_bias);
    FREE(conv01_filter);
    conv01.get_mem(dn, dc, dh, dw);
    // prelu01
    INIT_PRELU();
    PreluLayer prelu01(dn, dc, dh, dw, conv01_bias);
    FREE(conv01_bias);
    // pool0
    MaxPoolingLayer pool0(dn, dc, dh, dw, 0, 0, 2, 2, 2, 2);
    pool0.get_mem(dn, dc, dh, dw);
    // conv11
    fn = 16; fc = dc; fh = 3; fw = 3;
    ph = fh / 2; pw = fw / 2; sh = 1;   sw = 1;
    INIT_CONV();
    ConvLayer conv11(dn, dc, dh, dw, ph, pw, sh, sw);
    conv11.init_weight(fn, fh, fw, conv01_filter, conv01_bias);
    FREE(conv01_bias);
    FREE(conv01_filter);
    conv11.get_mem(dn, dc, dh, dw);
    // prelu11
    INIT_PRELU();
    PreluLayer prelu11(dn, dc, dh, dw, conv01_bias);
    FREE(conv01_bias);
    // conv12
    fn = 16; fc = dc; fh = 3; fw = 3;
    ph = fh / 2; pw = fw / 2; sh = 1;   sw = 1;
    INIT_CONV();
    ConvLayer conv12(dn, dc, dh, dw, ph, pw, sh, sw);
    conv12.init_weight(fn, fh, fw, conv01_filter, conv01_bias);
    FREE(conv01_bias);
    FREE(conv01_filter);
    conv12.get_mem(dn, dc, dh, dw);
    // prelu12
    INIT_PRELU();
    PreluLayer prelu12(dn, dc, dh, dw, conv01_bias);
    FREE(conv01_bias);
    // conv13
    fn = 32; fc = dc; fh = 3; fw = 3;
    ph = fh / 2; pw = fw / 2; sh = 1;   sw = 1;
    INIT_CONV();
    ConvLayer conv13(dn, dc, dh, dw, ph, pw, sh, sw);
    conv13.init_weight(fn, fh, fw, conv01_filter, conv01_bias);
    FREE(conv01_bias);
    FREE(conv01_filter);
    conv13.get_mem(dn, dc, dh, dw);
    // prelu13
    INIT_PRELU();
    PreluLayer prelu13(dn, dc, dh, dw, conv01_bias);
    FREE(conv01_bias);
    // conv14
    fn = 32; fc = dc; fh = 3; fw = 3;
    ph = fh / 2; pw = fw / 2; sh = 1;   sw = 1;
    INIT_CONV();
    ConvLayer conv14(dn, dc, dh, dw, ph, pw, sh, sw);
    conv14.init_weight(fn, fh, fw, conv01_filter, conv01_bias);
    FREE(conv01_bias);
    FREE(conv01_filter);
    conv14.get_mem(dn, dc, dh, dw);
    // prelu14
    INIT_PRELU();
    PreluLayer prelu14(dn, dc, dh, dw, conv01_bias);
    FREE(conv01_bias);
    // pool1
    MaxPoolingLayer pool1(dn, dc, dh, dw, 0, 0, 2, 2, 2, 2);
    pool1.get_mem(dn, dc, dh, dw);
    
    // conv21
    fn = 48; fc = dc; fh = 3; fw = 3;
    ph = fh / 2; pw = fw / 2; sh = 1;   sw = 1;
    INIT_CONV();
    ConvLayer conv21(dn, dc, dh, dw, ph, pw, sh, sw);
    conv21.init_weight(fn, fh, fw, conv01_filter, conv01_bias);
    FREE(conv01_bias);
    FREE(conv01_filter);
    conv21.get_mem(dn, dc, dh, dw);
    // prelu21
    INIT_PRELU();
    PreluLayer prelu21(dn, dc, dh, dw, conv01_bias);
    FREE(conv01_bias);
    // conv22
    fn = 48; fc = dc; fh = 3; fw = 3;
    ph = fh / 2; pw = fw / 2; sh = 1;   sw = 1;
    INIT_CONV();
    ConvLayer conv22(dn, dc, dh, dw, ph, pw, sh, sw);
    conv22.init_weight(fn, fh, fw, conv01_filter, conv01_bias);
    FREE(conv01_bias);
    FREE(conv01_filter);
    conv22.get_mem(dn, dc, dh, dw);
    // prelu22
    INIT_PRELU();
    PreluLayer prelu22(dn, dc, dh, dw, conv01_bias);
    FREE(conv01_bias);
    // conv23
    fn = 64; fc = dc; fh = 3; fw = 3;
    ph = fh / 2; pw = fw / 2; sh = 1;   sw = 1;
    INIT_CONV();
    ConvLayer conv23(dn, dc, dh, dw, ph, pw, sh, sw);
    conv23.init_weight(fn, fh, fw, conv01_filter, conv01_bias);
    FREE(conv01_bias);
    FREE(conv01_filter);
    conv23.get_mem(dn, dc, dh, dw);
    // prelu23
    INIT_PRELU();
    PreluLayer prelu23(dn, dc, dh, dw, conv01_bias);
    FREE(conv01_bias);
    // conv24
    fn = 64; fc = dc; fh = 3; fw = 3;
    ph = fh / 2; pw = fw / 2; sh = 1;   sw = 1;
    INIT_CONV();
    ConvLayer conv24(dn, dc, dh, dw, ph, pw, sh, sw);
    conv24.init_weight(fn, fh, fw, conv01_filter, conv01_bias);
    FREE(conv01_bias);
    FREE(conv01_filter);
    conv24.get_mem(dn, dc, dh, dw);
    // prelu24
    INIT_PRELU();
    PreluLayer prelu24(dn, dc, dh, dw, conv01_bias);
    FREE(conv01_bias);
    // conv25
    fn = 80; fc = dc; fh = 3; fw = 3;
    ph = fh / 2; pw = fw / 2; sh = 1;   sw = 1;
    INIT_CONV();
    ConvLayer conv25(dn, dc, dh, dw, ph, pw, sh, sw);
    conv25.init_weight(fn, fh, fw, conv01_filter, conv01_bias);
    FREE(conv01_bias);
    FREE(conv01_filter);
    conv25.get_mem(dn, dc, dh, dw);
    // prelu25
    INIT_PRELU();
    PreluLayer prelu25(dn, dc, dh, dw, conv01_bias);
    FREE(conv01_bias);
    // conv26
    fn = 80; fc = dc; fh = 3; fw = 3;
    ph = fh / 2; pw = fw / 2; sh = 1;   sw = 1;
    INIT_CONV();
    ConvLayer conv26(dn, dc, dh, dw, ph, pw, sh, sw);
    conv26.init_weight(fn, fh, fw, conv01_filter, conv01_bias);
    FREE(conv01_bias);
    FREE(conv01_filter);
    conv26.get_mem(dn, dc, dh, dw);
    // prelu26
    INIT_PRELU();
    PreluLayer prelu26(dn, dc, dh, dw, conv01_bias);
    FREE(conv01_bias);
    // pool2
    MaxPoolingLayer pool2(dn, dc, dh, dw, 0, 0, 2, 2, 2, 2);
    pool2.get_mem(dn, dc, dh, dw);
    
    // conv31
    fn = 160; fc = dc; fh = 3; fw = 3;
    ph = fh / 2; pw = fw / 2; sh = 1;   sw = 1;
    INIT_CONV();
    ConvLayer conv31(dn, dc, dh, dw, ph, pw, sh, sw);
    conv31.init_weight(fn, fh, fw, conv01_filter, conv01_bias);
    FREE(conv01_bias);
    FREE(conv01_filter);
    conv31.get_mem(dn, dc, dh, dw);
    // prelu31
    INIT_PRELU();
    PreluLayer prelu31(dn, dc, dh, dw, conv01_bias);
    FREE(conv01_bias);
    // conv32
    fn = 160; fc = dc; fh = 3; fw = 3;
    ph = fh / 2; pw = fw / 2; sh = 1;   sw = 1;
    INIT_CONV();
    ConvLayer conv32(dn, dc, dh, dw, ph, pw, sh, sw);
    conv32.init_weight(fn, fh, fw, conv01_filter, conv01_bias);
    FREE(conv01_bias);
    FREE(conv01_filter);
    conv32.get_mem(dn, dc, dh, dw);
    // prelu32
    INIT_PRELU();
    PreluLayer prelu32(dn, dc, dh, dw, conv01_bias);
    FREE(conv01_bias);
    // conv33
    fn = 160; fc = dc; fh = 3; fw = 3;
    ph = fh / 2; pw = fw / 2; sh = 1;   sw = 1;
    INIT_CONV();
    ConvLayer conv33(dn, dc, dh, dw, ph, pw, sh, sw);
    conv33.init_weight(fn, fh, fw, conv01_filter, conv01_bias);
    FREE(conv01_bias);
    FREE(conv01_filter);
    conv33.get_mem(dn, dc, dh, dw);
    // prelu33
    INIT_PRELU();
    PreluLayer prelu33(dn, dc, dh, dw, conv01_bias);
    FREE(conv01_bias);
    // conv34
    fn = 160; fc = dc; fh = 3; fw = 3;
    ph = fh / 2; pw = fw / 2; sh = 1;   sw = 1;
    INIT_CONV();
    ConvLayer conv34(dn, dc, dh, dw, ph, pw, sh, sw);
    conv34.init_weight(fn, fh, fw, conv01_filter, conv01_bias);
    FREE(conv01_bias);
    FREE(conv01_filter);
    conv34.get_mem(dn, dc, dh, dw);
    // prelu34
    INIT_PRELU();
    PreluLayer prelu34(dn, dc, dh, dw, conv01_bias);
    FREE(conv01_bias);
    // pool3
    MaxPoolingLayer pool3(dn, dc, dh, dw, 0, 0, 2, 2, 2, 2);
    pool3.get_mem(dn, dc, dh, dw);
    
    // conv41
    fn = 64; fc = dc; fh = 5; fw = 5;
    ph = fh / 2; pw = fw / 2; sh = 1;   sw = 1;
    INIT_CONV();
    ConvLayer conv41(dn, dc, dh, dw, ph, pw, sh, sw);
    conv41.init_weight(fn, fh, fw, conv01_filter, conv01_bias);
    FREE(conv01_bias);
    FREE(conv01_filter);
    conv41.get_mem(dn, dc, dh, dw);
    // prelu41
    INIT_PRELU();
    PreluLayer prelu41(dn, dc, dh, dw, conv01_bias);
    FREE(conv01_bias);

    // -------------------cnn net start----------------------------
    conv01.forward(d_conv01_data);
    prelu01.forward(conv01.top_);
    pool0.forward(conv01.top_);
    conv11.forward(pool0.pool_);
    prelu11.forward(conv11.top_);
    conv12.forward(conv11.top_);
    prelu12.forward(conv12.top_);
    conv13.forward(conv12.top_);
    prelu13.forward(conv13.top_);
    conv14.forward(conv13.top_);
    prelu14.forward(conv14.top_);
    pool1.forward(conv14.top_);

    conv21.forward(pool1.pool_);
    prelu21.forward(conv21.top_);
    conv22.forward(conv21.top_);
    prelu22.forward(conv22.top_);
    conv23.forward(conv22.top_);
    prelu23.forward(conv23.top_);
    conv24.forward(conv23.top_);
    prelu24.forward(conv24.top_);
    conv25.forward(conv24.top_);
    prelu25.forward(conv25.top_);
    conv26.forward(conv25.top_);
    prelu26.forward(conv26.top_);
    pool2.forward(conv26.top_);

    conv31.forward(pool2.pool_);
    prelu31.forward(conv31.top_);
    conv32.forward(conv31.top_);
    prelu32.forward(conv32.top_);
    conv33.forward(conv32.top_);
    prelu33.forward(conv33.top_);
    conv34.forward(conv33.top_);
    prelu34.forward(conv34.top_);
    pool3.forward(conv34.top_);

    conv41.forward(pool3.pool_);
    prelu41.forward(conv41.top_);

    //cl_mem d_conv01_out;
    //conv01.get_mem(d_conv01_out, dn, dc, dh, dw);
    float *h_out = (float *)alignedMalloc(sizeof(float) * MUL_4(dn, dc, dh, dw));
    status = clEnqueueReadBuffer(queues[K_IM2COL], conv41.top_, CL_TRUE, 0, sizeof(float) * MUL_4(dn, dc, dh, dw), h_out, 0, NULL, NULL);
    clFinish(queues[K_IM2COL]);
    
    WRITE_BIN(h_out, "prelu41-out-fpga.bin", MUL_4(dn, dc, dh, dw)); 
    
    if(conv01_data)     alignedFree(conv01_data);
    if(h_out)           alignedFree(h_out);
    if(conv01_filter)   alignedFree(conv01_filter);
    if(d_conv01_data)   clReleaseMemObject(d_conv01_data);
    //if(d_conv01_out)    clReleaseMemObject(d_conv01_out);
}

void attribute_test(void) {
    int offset = 0, ret = 0;
    FILE        *fp_data, *fp_model;
    fp_data     = fopen("conv01-in.bin", "rb");
    fp_model    = fopen("attribute-model.bin", "rb");


    // Convolution layer 1
    blob_shape_t conv01_data(1, 3, 256, 256);
	conv01_data.data = (float *)alignedMalloc(size_blob(conv01_data)*sizeof(float));
	ret = fread(conv01_data.data, sizeof(float), size_blob(conv01_data), fp_data);

    blob_shape_t conv01_filter(16, conv01_data.channels, 7, 7);
	conv01_filter.data = (float *)alignedMalloc(size_blob(conv01_filter)*sizeof(float));
	
    conv_param_t conv01_param(conv01_filter.height/2, conv01_filter.width/2, 2, 2);
	
    blob_shape_t conv01_out(1, conv01_filter.num, 
            (conv01_data.height + 2 * conv01_param.pad_h - conv01_filter.height) / conv01_param.stride_h + 1,
            (conv01_data.width + 2 * conv01_param.pad_w - conv01_filter.width) / conv01_param.stride_w + 1
            );
	conv01_out.data = (float *)alignedMalloc(size_blob(conv01_out)*sizeof(float));

	float *conv01_bias = (float *)alignedMalloc(conv01_out.channels * sizeof(float));

	load_model_param(fp_model, offset, size_blob(conv01_filter), conv01_filter.data);
	offset += size_blob(conv01_filter);
	load_model_param(fp_model, offset, conv01_out.channels, conv01_bias);
	offset += conv01_out.channels;
	
    pfunc_conv(conv01_data, conv01_filter, conv01_out, conv01_param);
	pfunc_bias(conv01_out.data, conv01_out.channels, conv01_out.height*conv01_out.width, conv01_bias);
    // ---------
    WRITE_BIN(conv01_out.data, "conv01-out-sim.bin", conv01_out.height * conv01_out.width * conv01_out.channels); 

    alignedFree(conv01_data.data);
    alignedFree(conv01_filter.data);
    alignedFree(conv01_bias);
    alignedFree(conv01_out.data);
}

void lenet_test(int file_sel) {

	blob_shape_t conv1_data, conv1_filter, conv1_out;
	conv_param_t conv1_param;
	float        *conv1_bias;
	uint         offset = 0;
	blob_shape_t pool1_out;
	conv_param_t pool1_param;
	blob_shape_t conv2_filter, conv2_out;
	conv_param_t conv2_param;
	float        *conv2_bias;
	blob_shape_t pool2_out;
	conv_param_t pool2_param;
	float        *inner1_weight, *inner1_bias, *inner1_out;
	float        *relu1_out;
	float        *inner2_weight, *inner2_bias, *inner2_out;
	float        *soft1_out;
	FILE         *fp_data, *fp_model;

	fp_data     = fopen("data-all.bin", "rb");
	fp_model    = fopen("model.bin", "rb");
	
    // Convolution layer 1
	// num:20, kernel:5, stride:1
	// parameter, weight: 5*5*20; bias:20
	conv1_data.num = 1;
	conv1_data.channels = 1;
	conv1_data.height = 28;
	conv1_data.width = 28;
	conv1_data.data = (float *)alignedMalloc(conv1_data.num*conv1_data.channels*conv1_data.height*conv1_data.width*sizeof(float));
	// =============================================================
	// TODO: load data in
	fseek(fp_data, sizeof(float)*file_sel*28*28, 0);
	fread(conv1_data.data, sizeof(float), conv1_data.num*conv1_data.channels*conv1_data.height*conv1_data.width, fp_data);
	// =============================================================

	conv1_filter.num = 20;
	conv1_filter.channels = 1;
	conv1_filter.height = 5;
	conv1_filter.width = 5;
	conv1_filter.data = (float *)alignedMalloc(conv1_filter.num*conv1_filter.channels*conv1_filter.height*conv1_filter.width*sizeof(float));
	conv1_param.pad_w = 0;
	conv1_param.pad_h = 0;
	conv1_param.stride_w = 1;
	conv1_param.stride_h = 1;
	conv1_out.num = 1;
	conv1_out.channels = 20;
	conv1_out.height = (conv1_data.height + 2 * conv1_param.pad_h - conv1_filter.height) / conv1_param.stride_h + 1;
	conv1_out.width = (conv1_data.width + 2 * conv1_param.pad_w - conv1_filter.width) / conv1_param.stride_w + 1;
	conv1_out.data = (float *)alignedMalloc(conv1_out.num*conv1_out.channels*conv1_out.height*conv1_out.width*sizeof(float));

	conv1_bias = (float *)alignedMalloc(20 * sizeof(float));

	load_model_param(fp_model, offset, 5 * 5 * 20, conv1_filter.data);
	offset += 5 * 5 * 20;
	load_model_param(fp_model, offset, 20, conv1_bias);
	offset += 20;
	// 20X(5*5), 25X(24*24)
	pfunc_conv(conv1_data, conv1_filter, conv1_out, conv1_param);
	pfunc_bias(conv1_out.data, conv1_out.channels, conv1_out.height*conv1_out.width, conv1_bias);

	// max-pooling layer 1
	// kernel: 2, stride: 2
	pool1_param.pad_w = 0;
	pool1_param.pad_h = 0;
	pool1_param.stride_w = 2;
	pool1_param.stride_h = 2;
	pool1_out.num = 1;
	pool1_out.channels = 20;
	pool1_out.height = (conv1_out.height + 2 * pool1_param.pad_h - pool1_param.stride_h) / pool1_param.stride_h + 1;
	pool1_out.width = (conv1_out.width + 2 * pool1_param.pad_w - pool1_param.stride_w) / pool1_param.stride_w + 1;
	pool1_out.data = (float *)alignedMalloc(pool1_out.num*pool1_out.channels*pool1_out.height*pool1_out.width*sizeof(float));
	pfunc_pooling(conv1_out, pool1_param, pool1_param.stride_h, pool1_param.stride_w, pool1_out);

	// Convolution layer 2
	// num:50, kernel:5, stride:1
	// parameter, weight: 5*5*20; bias:50
	conv2_filter.num = 50+2;
	conv2_filter.channels = 20;
	conv2_filter.height = 5;
	conv2_filter.width = 5;
	conv2_filter.data = (float *)alignedMalloc((conv2_filter.num)*conv2_filter.channels*conv2_filter.height*conv2_filter.width*sizeof(float));
	conv2_param.pad_w = 0;
	conv2_param.pad_h = 0;
	conv2_param.stride_w = 1;
	conv2_param.stride_h = 1;
	conv2_out.num = 1;
	conv2_out.channels = 50;
	conv2_out.height = (pool1_out.height + 2 * conv2_param.pad_h - conv2_filter.height) / conv2_param.stride_h + 1;
	conv2_out.width = (pool1_out.width + 2 * conv2_param.pad_w - conv2_filter.width) / conv2_param.stride_w + 1;
	conv2_out.data = (float *)alignedMalloc(conv2_out.num*(conv2_out.channels+2)*conv2_out.height*conv2_out.width*sizeof(float));

	conv2_bias = (float *)alignedMalloc(50 * sizeof(float));

	load_model_param(fp_model, offset, 5 * 5 * 20 * 50, conv2_filter.data);
	offset += 5 * 5 * 20 * 50;
	load_model_param(fp_model, offset, 50, conv2_bias);
	offset += 50;
	pfunc_conv(pool1_out, conv2_filter, conv2_out, conv2_param);
	pfunc_bias(conv2_out.data, conv2_out.channels, conv2_out.height*conv2_out.width, conv2_bias);

	// max-pooling layer 2
	// kernel: 2, stride: 2
	pool2_param.pad_w = 0;
	pool2_param.pad_h = 0;
	pool2_param.stride_w = 2;
	pool2_param.stride_h = 2;
	pool2_out.num = 1;
	pool2_out.channels = 50;
	pool2_out.height = (conv2_out.height + 2 * pool2_param.pad_h - pool2_param.stride_h) / pool2_param.stride_h + 1;
	pool2_out.width = (conv2_out.width + 2 * pool2_param.pad_w - pool2_param.stride_w) / pool2_param.stride_w + 1;
	pool2_out.data = (float *)alignedMalloc(pool2_out.num*pool2_out.channels*pool2_out.height*pool2_out.width*sizeof(float));
	pfunc_pooling(conv2_out, pool2_param, pool2_param.stride_h, pool2_param.stride_w, pool2_out);

	// Inner-product layer 1
	// num: 500
	inner1_out = (float *)alignedMalloc(500 * 1 * sizeof(float));
	inner1_weight = (float *)alignedMalloc(500 * 50 * pool2_out.height*pool2_out.width*sizeof(float));
	inner1_bias = (float *)alignedMalloc(500 * sizeof(float));
	load_model_param(fp_model, offset, 500 * 50 * pool2_out.height*pool2_out.width, inner1_weight);
	offset += 500 * 50 * pool2_out.height*pool2_out.width;
	load_model_param(fp_model, offset, 500, inner1_bias);
	offset += 500;
	pfunc_inner_product(pool2_out.data, inner1_weight, inner1_bias, inner1_out, 500, 50 * pool2_out.height*pool2_out.width);

	// ReLu layer 1
	relu1_out = (float *)alignedMalloc(500 * sizeof(float));
	pfunc_relu(inner1_out, relu1_out, 500);

	// Inner-product layer 2
	// num: 10
	inner2_out = (float *)alignedMalloc(10 * 1 * sizeof(float));
	inner2_weight = (float *)alignedMalloc(10 * 500 * sizeof(float));
	inner2_bias = (float *)alignedMalloc(10 * 1 * sizeof(float));
	load_model_param(fp_model, offset, 10 * 500, inner2_weight);
	offset += 10 * 500;
	load_model_param(fp_model, offset, 10, inner2_bias);
	offset += 10;
	pfunc_inner_product(relu1_out, inner2_weight, inner2_bias, inner2_out, 10, 500);

	// Softmax layer 1
	soft1_out = (float *)alignedMalloc(10 * 1 * sizeof(float));
	pfunc_softmax(inner2_out, soft1_out, 10);

	printf("\n");
	printf("soft1_out output value\n");
	for (offset = 0; offset < 10; offset++)
		printf("%.8f  ", soft1_out[offset]);

	printf("\n");

	fclose(fp_data);
	fclose(fp_model);

	free(conv1_data.data);
	free(conv1_filter.data);
	free(conv1_out.data);
	free(conv1_bias);

	free(pool1_out.data);

	free(conv2_filter.data);
	free(conv2_out.data);
	free(conv2_bias);

	free(pool2_out.data);

	free(inner1_weight);
	free(inner1_bias);
	free(inner1_out);

	free(relu1_out);

	free(inner2_weight);
	free(inner2_bias);
	free(inner2_out);

	free(soft1_out);
}
#ifdef BIAS
void ocl_conv_bias(float * NO_ALIAS data, uint row, uint col, float * NO_ALIAS bias) {
    printf("entering func:\t%s\n", __FUNCTION__);

    cl_mem d_data   = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * col * row, NULL, &status);
    checkError(status, "Failed to allocate input device buffer\n");
    cl_mem d_bias   = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * row, NULL, &status);
    checkError(status, "Failed to allocate input device buffer\n");
    
    status = clEnqueueWriteBuffer(queues[K_BIAS], d_data, CL_TRUE, 0, sizeof(float) * col * row, data, 0, NULL, NULL);
    checkError(status, "Failed to copy data to device");
    status = clEnqueueWriteBuffer(queues[K_BIAS], d_bias, CL_TRUE, 0, sizeof(float) * row, bias, 0, NULL, NULL);
    checkError(status, "Failed to copy data to device");
    
    status = clSetKernelArg(kernels[K_BIAS], 0, sizeof(cl_mem), (void*)&d_data);
    checkError(status, "Failed to set im2col arg 0");
    status = clSetKernelArg(kernels[K_BIAS], 1, sizeof(cl_mem), (void*)&d_bias);
    checkError(status, "Failed to set im2col arg 1");
    status = clSetKernelArg(kernels[K_BIAS], 2, sizeof(int), (void*)&col);
    checkError(status, "Failed to set im2col arg 2");
    status = clFinish(queues[K_BIAS]);
    
    size_t channel_ext = row;
    cl_event event_bias;
    status = clEnqueueNDRangeKernel(queues[K_BIAS], kernels[K_BIAS], 1, NULL, &channel_ext, NULL, 0, NULL, &event_bias);
    clWaitForEvents(1, &event_bias);

    status = clEnqueueReadBuffer(queues[K_BIAS], d_data, CL_TRUE, 0, sizeof(float) * row * col, data, 0, NULL, NULL);
    clFinish(queues[K_BIAS]);

    if(event_bias)  clReleaseEvent(event_bias);
    if(d_data)      clReleaseMemObject(d_data);
    if(d_bias)      clReleaseMemObject(d_bias);
}
#endif

#ifdef SOFTMAX
void ocl_softmax_layer(float * NO_ALIAS in_data, float * NO_ALIAS out_data, uint num) {
    printf("entering func:\t%s\n", __FUNCTION__);

}
#endif

#ifdef RELU
void ocl_relu_layer(float * NO_ALIAS in_data, float * NO_ALIAS out_data, uint num) {
    printf("entering func:\t%s\n", __FUNCTION__);
    cl_mem d_data   = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * num, NULL, &status);
    checkError(status, "Failed to allocate input device buffer\n");
    cl_mem d_relu   = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * num, NULL, &status);
    checkError(status, "Failed to allocate input device buffer\n");

    status = clEnqueueWriteBuffer(queues[K_RELU], d_data, CL_TRUE, 0, sizeof(float) * num, in_data, 0, NULL, NULL);
    checkError(status, "Failed to copy data to device");
    
    status = clSetKernelArg(kernels[K_RELU], 0, sizeof(cl_mem), (void*)&d_data);
    checkError(status, "Failed to set im2col arg 0");
    status = clSetKernelArg(kernels[K_RELU], 1, sizeof(cl_mem), (void*)&d_relu);
    checkError(status, "Failed to set im2col arg 1");
    status = clFinish(queues[K_RELU]);
    
    size_t channel_ext = num;
    cl_event event_relu;
    status = clEnqueueNDRangeKernel(queues[K_RELU], kernels[K_RELU], 1, NULL, &channel_ext, NULL, 0, NULL, &event_relu);
    clWaitForEvents(1, &event_relu);

    status = clEnqueueReadBuffer(queues[K_RELU], d_relu, CL_TRUE, 0, sizeof(float) * num, out_data, 0, NULL, NULL);
    clFinish(queues[K_RELU]);

    if(event_relu)  clReleaseEvent(event_relu);
    if(d_data)      clReleaseMemObject(d_data);
    if(d_relu)      clReleaseMemObject(d_relu);
}
#endif

#ifdef INNER_PRODUCT
void ocl_inner_product_layer(float * NO_ALIAS in_data, float * NO_ALIAS in_weight, float * NO_ALIAS in_bias, float * NO_ALIAS out_data, uint row, uint col) {
    printf("entering func:\t%s\n", __FUNCTION__);
    cl_mem d_data   = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * col, NULL, &status);
    checkError(status, "Failed to allocate input device buffer\n");
    cl_mem d_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * col * row, NULL, &status);
    checkError(status, "Failed to allocate input device buffer\n");
    cl_mem d_bias   = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * row, NULL, &status);
    checkError(status, "Failed to allocate input device buffer\n");
    cl_mem d_out    = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * row, NULL, &status);
    checkError(status, "Failed to allocate input device buffer\n");

    status = clEnqueueWriteBuffer(queues[K_INNER_PRODUCT], d_data, CL_TRUE, 0, sizeof(float) * col, in_data, 0, NULL, NULL);
    checkError(status, "Failed to copy data to device");
    status = clEnqueueWriteBuffer(queues[K_INNER_PRODUCT], d_weight, CL_TRUE, 0, sizeof(float) * col * row, in_weight, 0, NULL, NULL);
    checkError(status, "Failed to copy data to device");
    status = clEnqueueWriteBuffer(queues[K_INNER_PRODUCT], d_bias, CL_TRUE, 0, sizeof(float) * row, in_bias, 0, NULL, NULL);
    checkError(status, "Failed to copy data to device");
    status = clFinish(queues[K_INNER_PRODUCT]);
    
    status = clSetKernelArg(kernels[K_INNER_PRODUCT], 0, sizeof(cl_mem), (void*)&d_data);
    checkError(status, "Failed to set im2col arg 0");
    status = clSetKernelArg(kernels[K_INNER_PRODUCT], 1, sizeof(cl_mem), (void*)&d_weight);
    checkError(status, "Failed to set im2col arg 1");
    status = clSetKernelArg(kernels[K_INNER_PRODUCT], 2, sizeof(cl_mem), (void*)&d_bias);
    checkError(status, "Failed to set im2col arg 2");
    status = clSetKernelArg(kernels[K_INNER_PRODUCT], 3, sizeof(cl_mem), (void*)&d_out);
    checkError(status, "Failed to set im2col arg 3");
    status = clSetKernelArg(kernels[K_INNER_PRODUCT], 4, sizeof(int), (void*)&col);
    checkError(status, "Failed to set im2col arg 4");
    
    size_t channel_ext = row;
    cl_event event_inner_product;
    status = clEnqueueNDRangeKernel(queues[K_INNER_PRODUCT], kernels[K_INNER_PRODUCT], 1, NULL, &channel_ext, NULL, 0, NULL, &event_inner_product);
    clWaitForEvents(1, &event_inner_product);

    status = clEnqueueReadBuffer(queues[K_INNER_PRODUCT], d_out, CL_TRUE, 0, sizeof(float) * row, out_data, 0, NULL, NULL);
    clFinish(queues[K_INNER_PRODUCT]);

    if(event_inner_product) clReleaseEvent(event_inner_product);
    if(d_data)              clReleaseMemObject(d_data);
    if(d_weight)            clReleaseMemObject(d_weight);
    if(d_bias)              clReleaseMemObject(d_bias);
    if(d_out)               clReleaseMemObject(d_weight);
}
#endif

#if defined(IM2COL) && defined(GEMM)
void ocl_conv_layer(const blob_shape_t src, const blob_shape_t filter, blob_shape_t dst, const conv_param_t param) {
    printf("entering func:\t%s\n", __FUNCTION__);
    int num_channel = src.channels;
    int size_kernel = filter.height;
    int width_img   = src.width;
    int height_img  = src.height;
    int size_pad    = param.pad_h;
    int size_stride = param.stride_h;
    int width_col   = (width_img + 2 * size_pad - size_kernel) / size_stride + 1;
    int height_col  = (height_img + 2 * size_pad - size_kernel) / size_stride + 1;
    int offset_img  = width_img * height_img;
    int offset_col  = width_col * height_col;
    
    cl_mem d_img    = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * num_channel * offset_img, NULL, &status);
    checkError(status, "Failed to allocate input device buffer\n");
    cl_mem d_col    = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * num_channel * offset_col * size_kernel * size_kernel, NULL, &status);
    checkError(status, "Failed to allocate output device buffer\n");
    status = clEnqueueWriteBuffer(queues[K_IM2COL], d_img, CL_TRUE, 0, sizeof(float) * num_channel * offset_img, src.data, 0, NULL, NULL);
    checkError(status, "Failed to copy data to device");
    status = clFinish(queues[K_IM2COL]);
    status = clSetKernelArg(kernels[K_IM2COL], 0, sizeof(cl_mem), (void*)&d_img);
    checkError(status, "Failed to set im2col arg 0");
    status = clSetKernelArg(kernels[K_IM2COL], 1, sizeof(cl_mem), (void*)&d_col);
    checkError(status, "Failed to set im2col arg 1");
    status = clSetKernelArg(kernels[K_IM2COL], 2, sizeof(int), (void*)&offset_img);
    checkError(status, "Failed to set im2col arg 2");
    status = clSetKernelArg(kernels[K_IM2COL], 3, sizeof(int), (void*)&offset_col);
    checkError(status, "Failed to set im2col arg 3");
    status = clSetKernelArg(kernels[K_IM2COL], 4, sizeof(int), (void*)&height_img);
    checkError(status, "Failed to set im2col arg 4");
    status = clSetKernelArg(kernels[K_IM2COL], 5, sizeof(int), (void*)&width_img);
    checkError(status, "Failed to set im2col arg 5");
    status = clSetKernelArg(kernels[K_IM2COL], 6, sizeof(int), (void*)&height_col);
    checkError(status, "Failed to set im2col arg 6");
    status = clSetKernelArg(kernels[K_IM2COL], 7, sizeof(int), (void*)&width_col);
    checkError(status, "Failed to set im2col arg 7");
    status = clSetKernelArg(kernels[K_IM2COL], 8, sizeof(int), (void*)&size_kernel);
    checkError(status, "Failed to set im2col arg 8");
    status = clSetKernelArg(kernels[K_IM2COL], 9, sizeof(int), (void*)&size_pad);
    checkError(status, "Failed to set im2col arg 9");
    status = clSetKernelArg(kernels[K_IM2COL], 10, sizeof(int), (void*)&size_stride);
    checkError(status, "Failed to set im2col arg 10");
    
    size_t channel_ext = num_channel * size_kernel * size_kernel;
    cl_event event_im2col;
    status = clEnqueueNDRangeKernel(queues[K_IM2COL], kernels[K_IM2COL], 1, NULL, &channel_ext, NULL, 0, NULL, &event_im2col);
    clWaitForEvents(1, &event_im2col);
  
    int M = filter.num, K = filter.channels * filter.width * filter.height, N = dst.width * dst.height;
    printf("M = %d, K = %d, N = %d\n", M, K, N);
    size_t wg_size[2] = {1, 1};
    size_t g_size[2] = {N, M};
    cl_event event_gemm;
    cl_mem d_a, d_c;
    d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * M * K, NULL, &status);
    d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * M * N, NULL, &status);

    status = clEnqueueWriteBuffer(queues[K_GEMM], d_a, CL_TRUE, 0, sizeof(float) * M * K, filter.data, 0, NULL, NULL);

    status     = clSetKernelArg(kernels[K_GEMM], 0, sizeof(cl_mem), &d_c);
    status    |= clSetKernelArg(kernels[K_GEMM], 1, sizeof(cl_mem), &d_a);
    status    |= clSetKernelArg(kernels[K_GEMM], 2, sizeof(cl_mem), &d_col);
    status    |= clSetKernelArg(kernels[K_GEMM], 3, sizeof(unsigned int), &M);
    status    |= clSetKernelArg(kernels[K_GEMM], 4, sizeof(unsigned int), &K);
    status    |= clSetKernelArg(kernels[K_GEMM], 5, sizeof(unsigned int), &N);
    clFinish(queues[K_GEMM]);

    status = clEnqueueNDRangeKernel(queues[K_GEMM], kernels[K_GEMM], 2, NULL, g_size, wg_size, 0, NULL, &event_gemm);
    clWaitForEvents(1, &event_gemm);
    status = clEnqueueReadBuffer(queues[K_GEMM], d_c, CL_TRUE, 0, sizeof(float) * M * N, dst.data, 0, NULL, NULL);
    clFinish(queues[K_GEMM]);

    if(event_gemm)      clReleaseEvent(event_gemm);
    if(event_im2col)    clReleaseEvent(event_im2col);
    if(d_img)           clReleaseMemObject(d_img);
    if(d_col)           clReleaseMemObject(d_col);
    if(d_a)             clReleaseMemObject(d_a);
    if(d_c)             clReleaseMemObject(d_c);
}
#endif

#ifdef MAX_POOLING
void ocl_pooling_max_layer(const blob_shape_t src, const conv_param_t param, const int kernel_h, const int kernel_w, blob_shape_t dst) {
    printf("entering func:\t%s\n", __FUNCTION__);
    int size_pool       = dst.num * dst.channels * dst.height * dst.width;
    int size_input      = src.num * src.channels * src.height * src.width;
    cl_mem d_input      = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * size_input, NULL, &status);
    checkError(status, "Failed to allocate input device buffer\n");
    cl_mem d_pool       = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * size_pool, NULL, &status);
    checkError(status, "Failed to allocate size pool\n");
    status = clEnqueueWriteBuffer(queues[K_MAX_POOLING], d_input, CL_TRUE, 0, sizeof(float) * size_input, src.data, 0, NULL, NULL);
    checkError(status, "Failed to copy data to device");
    status = clFinish(queues[K_MAX_POOLING]);
    status = clSetKernelArg(kernels[K_MAX_POOLING], 0, sizeof(cl_mem), (void*)&d_input);
    status = clSetKernelArg(kernels[K_MAX_POOLING], 1, sizeof(cl_mem), (void*)&d_pool);
    status = clSetKernelArg(kernels[K_MAX_POOLING], 2, sizeof(int), (void*)&src.height);
    status = clSetKernelArg(kernels[K_MAX_POOLING], 3, sizeof(int), (void*)&src.width);
    status = clSetKernelArg(kernels[K_MAX_POOLING], 4, sizeof(int), (void*)&dst.height);
    status = clSetKernelArg(kernels[K_MAX_POOLING], 5, sizeof(int), (void*)&dst.width);
    int offset_input = src.height * src.width;
    status = clSetKernelArg(kernels[K_MAX_POOLING], 6, sizeof(int), (void*)&offset_input);
    status = clSetKernelArg(kernels[K_MAX_POOLING], 7, sizeof(int), (void*)&param.stride_w);
    status = clSetKernelArg(kernels[K_MAX_POOLING], 8, sizeof(int), (void*)&kernel_w);
    status = clSetKernelArg(kernels[K_MAX_POOLING], 9, sizeof(int), (void*)&param.pad_w);
    
    cl_event event_max_pooling;
    size_t channel_ext = dst.channels * dst.height * dst.width;

    status = clEnqueueNDRangeKernel(queues[K_MAX_POOLING], kernels[K_MAX_POOLING], 1, NULL, &channel_ext, NULL, 0, NULL, &event_max_pooling);
    clWaitForEvents(1, &event_max_pooling);

    status = clEnqueueReadBuffer(queues[K_MAX_POOLING], d_pool, CL_TRUE, 0, sizeof(float) * size_pool, dst.data, 0, NULL, NULL);
    status = clFinish(queues[K_MAX_POOLING]);
    if(d_input)             clReleaseMemObject(d_input);
    if(d_pool)              clReleaseMemObject(d_pool);
    if(event_max_pooling)   clReleaseEvent(event_max_pooling);
}
#endif

#ifdef IM2COL
bool im2col_test(void) {
    int num_channel = 128;
    int size_kernel = 3;
    int width_img   = 16;
    int height_img  = 16;
    int size_pad    = size_kernel / 2;
    int width_col   = width_img + 2 * size_pad - size_kernel + 1;
    int height_col  = height_img + 2 * size_pad - size_kernel + 1;
    int offset_img  = width_img * height_img;
    int offset_col  = width_col * height_col;
    float *h_img = (float *)alignedMalloc(sizeof(float) * num_channel * offset_img);
    float *h_col = (float *)alignedMalloc(sizeof(float) * num_channel * offset_col * size_kernel * size_kernel);
    float *h_ref = (float *)alignedMalloc(sizeof(float) * num_channel * offset_col * size_kernel * size_kernel);
    printf("channel num\t\t:%d\n", num_channel);
    printf("kernel size\t\t:%d\n", size_kernel);
    printf("image width\t\t:%d\n", width_img);
    printf("image height\t\t:%d\n", height_img);
    printf("padding size\t\t:%d\n", size_pad);

    for(int i = 0; i < num_channel; i++)
        for(int j = 0; j < height_img; j++)
            for(int k = 0; k < width_img; k++)
                h_img[i*offset_img+j*width_img+k] = rand() / (float)RAND_MAX;
#if 0
    for(int i = 0; i < num_channel; i++) {
        printf("----channel %d----\n", i);
        for(int j = 0; j < height_img; j++) {
            for(int k = 0; k < width_img; k++)
                printf("%3.1f\t", h_img[i*offset_img+j*width_img+k]);
            printf("\n");
        }
    }
#endif

    cl_mem d_img    = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * num_channel * offset_img, NULL, &status);
    checkError(status, "Failed to allocate input device buffer\n");
    cl_mem d_col    = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * num_channel * offset_col * size_kernel * size_kernel, NULL, &status);
    checkError(status, "Failed to allocate output device buffer\n");
    status = clEnqueueWriteBuffer(queues[K_IM2COL], d_img, CL_TRUE, 0, sizeof(float) * num_channel * offset_img, h_img, 0, NULL, NULL);
    checkError(status, "Failed to copy data to device");
    status = clFinish(queues[K_IM2COL]);
    status = clSetKernelArg(kernels[K_IM2COL], 0, sizeof(cl_mem), (void*)&d_img);
    checkError(status, "Failed to set im2col arg 0");
    status = clSetKernelArg(kernels[K_IM2COL], 1, sizeof(cl_mem), (void*)&d_col);
    checkError(status, "Failed to set im2col arg 1");
    status = clSetKernelArg(kernels[K_IM2COL], 2, sizeof(int), (void*)&offset_img);
    checkError(status, "Failed to set im2col arg 2");
    status = clSetKernelArg(kernels[K_IM2COL], 3, sizeof(int), (void*)&offset_col);
    checkError(status, "Failed to set im2col arg 3");
    status = clSetKernelArg(kernels[K_IM2COL], 4, sizeof(int), (void*)&height_img);
    checkError(status, "Failed to set im2col arg 4");
    status = clSetKernelArg(kernels[K_IM2COL], 5, sizeof(int), (void*)&width_img);
    checkError(status, "Failed to set im2col arg 5");
    status = clSetKernelArg(kernels[K_IM2COL], 6, sizeof(int), (void*)&height_col);
    checkError(status, "Failed to set im2col arg 6");
    status = clSetKernelArg(kernels[K_IM2COL], 7, sizeof(int), (void*)&width_col);
    checkError(status, "Failed to set im2col arg 7");
    status = clSetKernelArg(kernels[K_IM2COL], 8, sizeof(int), (void*)&size_kernel);
    checkError(status, "Failed to set im2col arg 8");
    status = clSetKernelArg(kernels[K_IM2COL], 9, sizeof(int), (void*)&size_pad);
    checkError(status, "Failed to set im2col arg 9");
    
    size_t channel_ext = num_channel * size_kernel * size_kernel;
    
    double time_cost = getCurrentTimestamp();
    double time_event = 0;
    double time_iter;
    cl_event event_im2col;
#if 0
    time_cost = 0.0;
    int i = 0;
    int temp        = i / size_kernel;
    int w_offset    = i - temp * size_kernel;
    int c_offset    = temp / size_kernel;
    int h_offset    = temp - c_offset * size_kernel;
    int img_inc     = c_offset * offset_img;
    int col_inc     = i * offset_col;
    i++;
    for(; i < channel_ext + 1; i++) { 
        status = clSetKernelArg(kernels[K_IM2COL], 2, sizeof(int), (void*)&img_inc);
        checkError(status, "Failed to set im2col arg 2");
        status = clSetKernelArg(kernels[K_IM2COL], 3, sizeof(int), (void*)&col_inc);
        checkError(status, "Failed to set im2col arg 3");
        status = clSetKernelArg(kernels[K_IM2COL], 10, sizeof(int), (void*)&w_offset);
        checkError(status, "Failed to set im2col arg 2");
        status = clSetKernelArg(kernels[K_IM2COL], 11, sizeof(int), (void*)&h_offset);
        checkError(status, "Failed to set im2col arg 3");
        //status = clEnqueueTask(queues[K_IM2COL], kernels[K_IM2COL], 0, NULL, NULL);
        //status = clFinish(queues[K_IM2COL]);
        
        time_iter = getCurrentTimestamp();
        status = clEnqueueTask(queues[K_IM2COL], kernels[K_IM2COL], 0, NULL, &event_im2col);
        clWaitForEvents(1, &event_im2col);
        time_cost += (getCurrentTimestamp() - time_iter);

        cl_ulong s_time, e_time;
        status = clGetEventProfilingInfo(event_im2col, CL_PROFILING_COMMAND_QUEUED, sizeof(s_time), &s_time, NULL); 
        status = clGetEventProfilingInfo(event_im2col, CL_PROFILING_COMMAND_END, sizeof(e_time), &e_time, NULL); 
        time_event += (double)(e_time - s_time) * 1e-6;
        
        temp        = i / size_kernel;
        w_offset    = i - temp * size_kernel;
        c_offset    = temp / size_kernel;
        h_offset    = temp - c_offset * size_kernel;
        img_inc     = c_offset * offset_img;
        col_inc     = i * offset_col;
    }
#else
    time_cost = getCurrentTimestamp();
    status = clEnqueueNDRangeKernel(queues[K_IM2COL], kernels[K_IM2COL], 1, NULL, &channel_ext, NULL, 0, NULL, &event_im2col);
    clWaitForEvents(1, &event_im2col);
    time_cost = getCurrentTimestamp() - time_cost;
    
    cl_ulong s_time, e_time;
    status = clGetEventProfilingInfo(event_im2col, CL_PROFILING_COMMAND_QUEUED, sizeof(s_time), &s_time, NULL); 
    status = clGetEventProfilingInfo(event_im2col, CL_PROFILING_COMMAND_END, sizeof(e_time), &e_time, NULL); 
    time_event = (double)(e_time - s_time) * 1e-6;
#endif
    
    status = clEnqueueReadBuffer(queues[K_IM2COL], d_col, CL_TRUE, 0, sizeof(float) * num_channel * offset_col * size_kernel * size_kernel, h_col, 0, NULL, NULL);
    status = clFinish(queues[K_IM2COL]);
    im2col_ref(h_img, h_ref, num_channel, height_img, width_img, size_kernel, size_kernel, size_pad, size_pad, 1, 1); 
   
#if 0
    for(int i = 0; i < num_channel * size_kernel * size_kernel; i++) {
        printf("----channel %d----\n", i);
        for(int j = 0; j < height_col; j++) {
            for(int k = 0; k < width_col; k++) {
                printf("%3.1f,%3.1f\t", h_col[i*offset_col+j*width_col+k], h_ref[i*offset_col+j*width_col+k]);
            }
            printf("\n");
        }
    }
#endif
    bool flag;
    int cnt_err = 0;
    for(int i = 0; i < num_channel * size_kernel * size_kernel; i++) {
        for(int j = 0; j < height_col; j++) {
            for(int k = 0; k < width_col; k++) {
                int offset = i*offset_col+j*width_col+k;
                flag = approximatelyEqual(h_ref[offset], h_col[offset], 0.01);
                if(false == flag) cnt_err++;
            }
        }
    }
    if(cnt_err == 0) printf("im2col success, time cost(host clock) is %5.3lf(ms), time event(device clock) is %5.3lf(ms)\n", time_cost * 1e-6, time_event);
    else             printf("im2col fail, time cost (host clock)is %5.3lf(ms), time event(device clock) is %5.3f(ms)\n", time_cost * 1e-6, time_event);
    
    if(h_img)   alignedFree(h_img);
    if(h_col)   alignedFree(h_col);
    if(h_ref)   alignedFree(h_ref);
    if(d_img)   clReleaseMemObject(d_img);
    if(d_col)   clReleaseMemObject(d_col);

    if(event_im2col)    clReleaseEvent(event_im2col);
    return true;
}
#endif

// Set up the context, device, kernels, and buffers...
bool init() {
    cl_int status;

    // Start everything at NULL to help identify errors
    for(int i = 0; i < K_NUM_KERNELS; ++i){
        kernels[i] = NULL;
        queues[i] = NULL;
    }

    // Locate files via. relative paths
    if(!setCwdToExeDir()) {
        return false;
    }

    // Get the OpenCL platform.
    platform = findPlatform("Altera");
    if(platform == NULL) {
        printf("ERROR: Unable to find Altera OpenCL platform\n");
        return false;
    }

    // Query the available OpenCL devices and just use the first device if we find
    // more than one
    scoped_array<cl_device_id> devices;
    cl_uint num_devices;
    devices.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));
    device = devices[0];

    // Create the context.
    context = clCreateContext(NULL, 1, &device, &oclContextCallback, NULL, &status);
    checkError(status, "Failed to create context");

    // Create the command queues
    for(int i=0; i<K_NUM_KERNELS; ++i) {
        queues[i] = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
        checkError(status, "Failed to create command queue (%d)", i);
    }

    // Create the program.
    std::string binary_file = getBoardBinaryFile("cnn", device);
    printf("Using AOCX: %s\n\n", binary_file.c_str());
    program = createProgramFromBinary(context, binary_file.c_str(), &device, 1);

    // Build the program that was just created.
    status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
    checkError(status, "Failed to build program");

    // Create the kernel - name passed in here must match kernel name in the
    // original CL file, that was compiled into an AOCX file using the AOC tool
    for(int i=0; i<K_NUM_KERNELS; ++i) {
        kernels[i] = clCreateKernel(program, kernel_names[i], &status);
        checkError(status, "Failed to create kernel (%d: %s)", i, kernel_names[i]);
    }

    return true;
}

// Free the resources allocated during initialization
void cleanup() {
    for(int i=0; i<K_NUM_KERNELS; ++i) {
        if(kernels[i]) 
        clReleaseKernel(kernels[i]);  
    }
    if(program) 
        clReleaseProgram(program);
    for(int i=0; i<K_NUM_KERNELS; ++i) {
        if(queues[i]) 
        clReleaseCommandQueue(queues[i]);
    }
    if(context) 
        clReleaseContext(context);
}

bool approximatelyEqual(float a, float b, float epsilon) {
    return fabs(a - b) <= ((fabs(a) < fabs(b) ? fabs(b) : fabs(a)) * epsilon);
}
