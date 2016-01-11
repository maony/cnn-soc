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
#include "channelizer_golden.h"
using namespace aocl_utils;

// The set of simultaneous kernels
enum KERNELS {
  K_READER,
  //K_FILTER,
  //K_REORDER,
  //K_FFT,
  K_COMPUTE,
  K_WRITER,
  K_NUM_KERNELS
};
static const char* kernel_names[K_NUM_KERNELS] =
{
#if 1
    "conv_read_16x16",
    "conv_compute_16x16",
    "conv_write_16x16"
#else
  "conv_read_8x8",
  //"filter",
  //"reorder",
  //"fft1d",
  "conv_compute_8x8",
  "conv_write_8x8"
#endif
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
static void test_channelizer(const int LOGN, const int P, const int PPC, const int ITERS);
static void test_channel_conv();

// Host memory buffers
float *h_inData, *h_outData;

// Device memory buffers
cl_mem d_inData, d_outData;

bool approximatelyEqual(float a, float b, float epsilon) {
    return fabs(a - b) <= ((fabs(a) < fabs(b) ? fabs(b) : fabs(a)) * epsilon);
}

// Entry point.
int main(int argc, char **argv) {
  Options options(argc, argv);

  // Parameters of our design - these must match up with the associated values in
  // the kernel
  const int LOGN = 12;
  const int P = 8;
  const int PPC = 8;

  int iters = P*1*1;
  if(options.has("i")) {
    iters = options.get<int>("i");
  }

  if(iters < (P)) {
    printf("Error: iters must be more than %d.\n", P);
    return 1;
  }
  else if((iters % P) != 0) {
    printf("Error: iters must be a multiple of P (%d).\n", P);
    return 1;
  }

  printf("Number of iterations is set to %d\n",iters);

  // Derived constants
  const int N = (1 << LOGN);
  const int M = (N*P);


  // Flush stdout immediately
  setbuf(stdout, NULL);

  // Setup the context, create the device and kernels...
  if(!init()) {
    return false;
  }
  printf("Init complete!\n");

  // Allocate host memory
  h_inData = (float *)alignedMalloc(sizeof(float) * M);
  h_outData = (float *)alignedMalloc(sizeof(float) * N);
  if (!(h_inData && h_outData)) {
    printf("ERROR: Couldn't create host buffers\n");
    return false;
  }

  // Test 4k point FFT transform
  // test_channelizer(LOGN, P, PPC, iters);
    
    test_channel_conv();

  // Free the resources allocated
  cleanup();
  return 0;
}

static void test_channel_conv() {
    printf("conv test\n");
    int input_size = 14;
    int filter_size = 3;
    int num_channel = 6;
    int num_filter = 128;
    int output_size = input_size + 2 * (filter_size / 2) - filter_size + 1;
    int input_offset = input_size + 2 * (filter_size / 2);
    int output_offset = input_offset; 
    int input_element = input_offset * input_offset * num_channel;
    int filter_element = filter_size * filter_size * num_channel * num_filter;
    int output_element = output_offset * output_offset * num_filter;
    
    printf("input size: %d\n", input_size);
    printf("padding size: %d\n", filter_size / 2);
    printf("input size(padding included): %d\n", input_offset);
    printf("input channel num: %d\n", num_channel);
    printf("input filter num: %d\n", num_filter);
    printf("output size: %d\n", output_size);

    float *h_in, *h_out, *h_filter, *h_ref;
    
    h_in    = (float *)alignedMalloc(sizeof(float) * input_offset * input_offset * num_channel);
    h_filter= (float *)alignedMalloc(sizeof(float) * filter_size * filter_size * num_channel * num_filter);
    h_out   = (float *)alignedMalloc(sizeof(float) * output_offset * output_offset * num_filter);
    h_ref   = (float *)alignedMalloc(sizeof(float) * output_offset * output_offset * num_filter);

    for(int i = 0; i < filter_element; i++) h_filter[i] = rand() / (float)RAND_MAX;
    for(int i = 0; i < output_element; i++) h_ref[i] = 0;
    for(int i = 0; i < input_element; i++) h_in[i] = 0;
    
    for(int i = 0; i < num_channel; i++)
        for(int j = 0; j < input_size; j++)
            for(int k = 0; k < input_size; k++)
                h_in[i*input_offset*input_offset + input_offset * (j+filter_size/2) + filter_size/2 + k] = rand() / (float)RAND_MAX;//i+j+k;

    float temp;
    for(int i = 0; i < num_filter; i++) {
        for(int j = 0; j < output_size; j++) {
            for(int k = 0; k < output_size; k++) {
                temp = 0.0;
                for(int m = 0; m < num_channel; m++) {
                    for(int n = 0; n < filter_size; n++) {
                        for(int r = 0; r < filter_size; r++) {
                            int offset_in = m*input_offset*input_offset + (j+n)*input_offset + k + r;
                            int offset_filter = i*num_channel*filter_size*filter_size + m*filter_size*filter_size + n*filter_size + r;
                            temp += h_in[offset_in] * h_filter[offset_filter];
                        }
                    }
                }
                h_ref[i * output_offset * output_offset + j * output_offset + k] = temp;
            }
        }
    }
#if 0
    for(int i = 0; i < num_channel; i++) {
        printf("-------------------input channel %d-------------------------\n", i);
        for(int j = 0; j < input_offset; j++) {
            for(int k = 0; k < input_offset; k++) {
                printf("%5.1f\t\t", h_in[i*input_offset*input_offset + input_offset * j + k]);
            }
            printf("\n");
        }
    }
   
    for(int m = 0; m < num_filter; m++) {
        printf("-------------------filter num %d-------------------------\n", m);
        for(int i = 0; i < num_channel; i++) {
            printf("-------------------filter channel %d-------------------------\n", i);
            for(int j = 0; j < filter_size; j++) {
                for(int k = 0; k < filter_size; k++) {
                    printf("%5.1f\t\t", h_filter[m * num_channel*filter_size*filter_size + i*filter_size*filter_size + filter_size * j + k]);
                }
                printf("\n");
            }
        }
    }
#endif
    cl_mem d_in, d_out, d_filter;
    d_in = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * input_element, NULL, &status);
    checkError(status, "Failed to allocate input device buffer\n");
    d_out = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * output_element, NULL, &status);
    checkError(status, "Failed to allocate output device buffer\n");
    d_filter = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * filter_element, NULL, &status);
    checkError(status, "Failed to allocate output device buffer\n");

    // Copy data from host to device
    status = clEnqueueWriteBuffer(queues[0], d_in, CL_TRUE, 0, sizeof(float) * input_element, h_in, 0, NULL, NULL);
    checkError(status, "Failed to copy data to device");
    status = clEnqueueWriteBuffer(queues[0], d_filter, CL_TRUE, 0, sizeof(float) * filter_element, h_filter, 0, NULL, NULL);
    checkError(status, "Failed to copy data to device");
    status = clFinish(queues[0]);

    int offset_filter, offset_out;
    
    status = clSetKernelArg(kernels[K_READER], 0, sizeof(cl_mem), (void*)&d_in);
    checkError(status, "Failed to set kernel_rd arg 0");
    status = clSetKernelArg(kernels[K_READER], 1, sizeof(unsigned int), (void*)&num_channel);
    checkError(status, "Failed to set kernel_rd arg 1");
    status = clSetKernelArg(kernels[K_COMPUTE], 0, sizeof(cl_mem), (void*)&d_filter);
    checkError(status, "Failed to set kernel_compute arg 0");
    status = clSetKernelArg(kernels[K_COMPUTE], 1, sizeof(unsigned int), (void*)&num_channel);
    checkError(status, "Failed to set kernel_compute arg 1");
    status = clSetKernelArg(kernels[K_WRITER], 0, sizeof(cl_mem), (void*)&d_out);
    checkError(status, "Failed to set kernel_write arg 0");
    
    double time, time_iter;
    
    for(unsigned int i = 0; i < num_filter; i++) {
        offset_filter = (i * num_channel * filter_size * filter_size);
        offset_out = (i * output_offset * output_offset) / 16;
        status = clSetKernelArg(kernels[K_COMPUTE], 2, sizeof(unsigned int), (void*)&offset_filter);
        checkError(status, "Failed to set kernel_compute arg 2");
        status = clSetKernelArg(kernels[K_WRITER], 1, sizeof(unsigned int), (void*)&offset_out);
        checkError(status, "Failed to set kernel_write arg 0");
       
        time_iter = getCurrentTimestamp();
        status = clEnqueueTask(queues[K_READER], kernels[K_READER], 0, NULL, NULL);
        status = clEnqueueTask(queues[K_COMPUTE], kernels[K_COMPUTE], 0, NULL, NULL);
        status = clEnqueueTask(queues[K_WRITER], kernels[K_WRITER], 0, NULL, NULL);

        status = clFinish(queues[K_READER]);
        status = clFinish(queues[K_COMPUTE]);
        status = clFinish(queues[K_WRITER]);
        time_iter = getCurrentTimestamp() - time_iter;
        time += time_iter;
    }
    status = clEnqueueReadBuffer(queues[0], d_out, CL_TRUE, 0, sizeof(float) * output_element, h_out, 0, NULL, NULL);
    
    printf("----------------------------------------------\n");
    double flops = (double)(2.0f * num_filter * output_size * output_size * num_channel * filter_size * filter_size / time);
    printf("kernel time cost is %6.3lf ms, %9.6lf GFLOPS\n", time * 1.0e-6, flops);
    printf("----------------------------------------------\n");
    printf("\n-------------------channel output-------------------------\n");
#if 1
#define EPSILON 0.01
    bool flag;
    int err_count = 0;

    for(int i = 0; i < num_filter; i++) {
#if 0
        printf("-------------------channel %d-------------------------\n", i);
#endif
        for(int j = 0; j < output_offset; j++) {
            for(int k = 0; k < output_offset; k++) {
                int offset = i*output_offset*output_offset + output_offset * j + k;
                flag = approximatelyEqual(h_ref[offset], h_out[offset], EPSILON);
                if(false == flag) err_count++;
#if 0
                printf("%5.1f, %5.1f\t", h_ref[offset], h_out[offset]);
#endif
            }
#if 0
            printf("\n");
#endif
        }
    }
#endif
    if(err_count > 0) printf("conv error %d\n", err_count);
    else              printf("conv ok\n");

    alignedFree(h_out);
    alignedFree(h_in);
    alignedFree(h_filter);
    alignedFree(h_ref);
    clReleaseMemObject(d_in);
    clReleaseMemObject(d_out);
    clReleaseMemObject(d_filter);
}

// The test harness - generate some data, feed it through the FPGA kernel and
// compare against a functionally equivalent implementation implemented in
// channelizer_golden.cpp.
void test_channelizer(const int LOGN, const int P, const int PPC, const int ITERS) {
  const int N = (1 << LOGN);
  const int M = (N*P);

  printf("Launching FFT transform\n");

  // Initialize input and produce verification data
  for (int i = 0; i < M; i++) {
    //h_inData[i] = (float)(cos( 128 * i * M_PI / (3*N)) 
    //              + cos(2048 * i * M_PI / (3*N))
    //              + 0.0001 * cos( 1765 * i * M_PI / (3*N)));
    h_inData[i] = (i < 25) ? i : i % 25;
  }

  // Create device buffers - assign the buffers in different banks for more efficient
  // memory access 
  d_inData = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * M, NULL, &status);
  checkError(status, "Failed to allocate input device buffer\n");
  d_outData = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_BANK_2_ALTERA, sizeof(float) * N, NULL, &status);
  checkError(status, "Failed to allocate output device buffer\n");

  // Copy data from host to device
  status = clEnqueueWriteBuffer(queues[0], d_inData, CL_TRUE, 0, sizeof(float) * M, h_inData, 0, NULL, NULL);
  checkError(status, "Failed to copy data to device");

  // Set the kernel arguments
  // Read
  status = clSetKernelArg(kernels[K_READER], 0, sizeof(cl_mem), (void*)&d_inData);
  checkError(status, "Failed to set kernel_rd arg 0");
  //// FFT
  //cl_uint iters = ITERS;
  //status = clSetKernelArg(kernels[K_FFT], 0, sizeof(cl_int), (void*)&iters);
  //checkError(status, "Failed to set kernel_ff arg 0");
  //// POLYPHASE
  //status = clSetKernelArg(kernels[K_FILTER], 0, sizeof(cl_int), (void*)&iters);
  //checkError(status, "Failed to set kernel_ff arg 0");
  // Write
  status = clSetKernelArg(kernels[K_WRITER], 0, sizeof(cl_mem), (void*)&d_outData);
  checkError(status, "Failed to set kernel_wr arg 0");

  // Get the timestamp to evaluate performance
  double time = getCurrentTimestamp();
  size_t window_size = N / PPC;
  size_t sample_size = window_size * ITERS;
  size_t wg_size[2] = {1, 1};
  size_t g_size[2] = {N, P};
  
  // READ
  status = clEnqueueTask(queues[K_READER], kernels[K_READER], 0, NULL, NULL);
  //status = clEnqueueNDRangeKernel(queues[K_READER], kernels[K_READER], 2, NULL, g_size, 
  //  wg_size, 0, NULL, NULL);
  //status = clEnqueueNDRangeKernel(queues[K_READER], kernels[K_READER], 1, NULL, 
  //  &sample_size, NULL, 0, NULL, NULL);
  checkError(status, "Failed to launch kernel_read");
  //// POLYPHASE
  //status = clEnqueueTask(queues[K_FILTER], kernels[K_FILTER], 0, NULL, NULL);
  //checkError(status, "Failed to launch kernel_read");
  // REORDER
  //status = clEnqueueNDRangeKernel(queues[K_REORDER], kernels[K_REORDER], 1, NULL, 
  //  &sample_size, &window_size, 0, NULL, NULL);
  //checkError(status, "Failed to launch kernel_reorder");
  //// FFT
  //status = clEnqueueTask(queues[K_FFT], kernels[K_FFT], 0, NULL, NULL);
  //checkError(status, "Failed to launch kernel_fft");
  /// Write
  status = clEnqueueTask(queues[K_WRITER], kernels[K_WRITER], 0, NULL, NULL);
  //status = clEnqueueNDRangeKernel(queues[K_WRITER], kernels[K_WRITER], 2, NULL, g_size, 
  //  wg_size, 0, NULL, NULL);
  //status = clEnqueueNDRangeKernel(queues[K_WRITER], kernels[K_WRITER], 1, NULL, 
  //  &sample_size, NULL, 0, NULL, NULL);
  checkError(status, "Failed to launch kernel_write");

  // Wait for command queue to complete pending events
  //for(int i=0; i<K_NUM_KERNELS; ++i) {
    status = clFinish(queues[K_READER]);
    //checkError(status, "Failed to finish (%d: %s)", i, kernel_names[i]);
  //}
    status = clFinish(queues[K_WRITER]);
    //checkError(status, "Failed to finish (%d: %s)", i, kernel_names[i]);

  // Record execution time
  time = getCurrentTimestamp() - time;

  // Copy results from device to host
  status = clEnqueueReadBuffer(queues[0], d_outData, CL_TRUE, 0, sizeof(float) * N, h_outData, 0, NULL, NULL);
  checkError(status, "Failed to copy data from device");
  printf("data out  1 \n");
  for(int i = 0; i < 52; i++) {
      printf("%5.3f, %5.3f\t", h_inData[i], h_outData[i]);
      // printf("%5.3f\t", h_outData[i]);
      if(i % 5 == 0 && i != 0)
          printf("\n");
  }

    printf("kernel twice\n");
    for(int i = 0; i < 50; i++)
        h_inData[i] = h_outData[i];

    status = clEnqueueWriteBuffer(queues[0], d_inData, CL_TRUE, 0, sizeof(float) * M, h_inData, 0, NULL, NULL);
    checkError(status, "Failed to copy data to device");
    //status = clSetKernelArg(kernels[K_READER], 0, sizeof(cl_mem), (void*)&d_inData);
    //checkError(status, "Failed to set kernel_rd arg 0");
    //status = clSetKernelArg(kernels[K_WRITER], 0, sizeof(cl_mem), (void*)&d_outData);
    //checkError(status, "Failed to set kernel_wr arg 0");
    status = clEnqueueTask(queues[K_READER], kernels[K_READER], 0, NULL, NULL);
    checkError(status, "Failed to launch kernel_read");
    status = clEnqueueTask(queues[K_WRITER], kernels[K_WRITER], 0, NULL, NULL);
    checkError(status, "Failed to launch kernel_write");
    status = clFinish(queues[K_READER]);
    status = clFinish(queues[K_WRITER]);
  
    status = clEnqueueReadBuffer(queues[0], d_outData, CL_TRUE, 0, sizeof(float) * N, h_outData, 0, NULL, NULL);
    checkError(status, "Failed to copy data from device");
    printf("data out  2\n");
    for(int i = 0; i < 52; i++) {
        printf("%5.3f, %5.3f\t", h_inData[i], h_outData[i]);
        // printf("%5.3f\t", h_outData[i]);
        if(i % 5 == 0 && i != 0)
            printf("\n");
    }
    printf("\n------------------------------\n");

  printf("\tProcessing time = %.4fms\n", (float)(time * 1E3));
  double gpoints_per_sec = ((double)(ITERS) * N / time) * 1E-9;
  printf("\tThroughput = %.4f Gpoints / sec\n", gpoints_per_sec);
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
  std::string binary_file = getBoardBinaryFile("channelizer", device);
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
  if(h_inData)
    alignedFree(h_inData);
  if (h_outData)
    alignedFree(h_outData);
  if (d_inData)
    clReleaseMemObject(d_inData);
  if (d_outData) 
    clReleaseMemObject(d_outData);
}



