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
using namespace aocl_utils;

// The set of simultaneous kernels
enum KERNELS {
  K_IM2COL,
  K_NUM_KERNELS
};
static const char* kernel_names[K_NUM_KERNELS] =
{
  "im2col"
};

// ACL runtime configuration
static cl_platform_id platform = NULL;
static cl_device_id device = NULL;
static cl_context context = NULL;
static cl_command_queue queues[K_NUM_KERNELS];
static cl_kernel kernels[K_NUM_KERNELS];
static cl_program program = NULL;
static cl_int status = 0;

// Function prototypes
bool init();
void cleanup();
bool approximatelyEqual(float a, float b, float epsilon);
bool im2col_test();
extern void ref_lenet(void);

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
    
    im2col_test();
    //ref_lenet();

    printf("--------------------------------------\n");
    printf("-----------test complete!-------------\n");

    // Free the resources allocated
    cleanup();
    return 0;
}

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
#if 1
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
        printf("time cost iter is %5.3lf\t\t", (getCurrentTimestamp() - time_iter) * 1e-6);

        cl_ulong s_time, e_time;
        status = clGetEventProfilingInfo(event_im2col, CL_PROFILING_COMMAND_QUEUED, sizeof(s_time), &s_time, NULL); 
        status = clGetEventProfilingInfo(event_im2col, CL_PROFILING_COMMAND_END, sizeof(e_time), &e_time, NULL); 
        time_event += (double)(e_time - s_time) * 1e-6;
        printf("time event is %5.3lf, total:all: %5.3lf\n", (e_time - s_time) * 1e-6, time_event);
        
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

    return true;
}

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
