#ifndef __LAYER__
#define __LAYER__
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#include "common.hpp"

using namespace aocl_utils;

extern cl_platform_id platform;
extern cl_device_id device;
extern cl_context context;
extern cl_command_queue queues[2];
extern cl_kernel kernels[K_NUM_KERNELS];
extern cl_program program;
extern cl_int status;

#if defined(IM2COL) && defined(GEMM)
class ConvLayer {
    public:
        ConvLayer(void);
        ~ConvLayer();
        ConvLayer(int dn, int dc, int dh, int dw, int ph, int pw, int sh, int sw);
        void init_weight(int fn, int fh, int fw, float *weight, float *bias);
        void get_mem(int &dn, int &dc, int &dh, int &dw);
        void forward(cl_mem bot);
    private:
        int n_bot_, c_bot_, h_bot_, w_bot_;
        int n_top_, c_top_, h_top_, w_top_;
        int ph_, pw_, sh_, sw_;
        int fn_, fc_, fh_, fw_;
        int offset_bot_, offset_top_;
        cl_mem weight_, bias_, col_;
        cl_ulong s_time, e_time;
        cl_event event_conv;
    public:
        double c_time;
        cl_mem top_;
};
#endif

#ifdef PRELU
class PreluLayer {
    public:
        PreluLayer();
        PreluLayer(int dn, int dc, int dh, int dw, float *slope);
        ~PreluLayer();
        void forward(cl_mem &data);
        double c_time;
    private:
        int n_, c_, h_, w_, size_;
        size_t g_size_;
        cl_mem slope_;
        cl_ulong s_time, e_time;
        cl_event event_prelu;
};
#endif

#ifdef MAX_POOLING
class MaxPoolingLayer {
    public:
        MaxPoolingLayer();
        MaxPoolingLayer(int dn, int dc, int dh, int dw, int ph, int pw, int sh, int sw, int kh, int kw);
        ~MaxPoolingLayer();
        void forward(cl_mem data);
        void get_mem(int &dn, int &dc, int &dh, int &dw);
        cl_mem pool_;
        double c_time;
    private:
        int n_bot_, c_bot_, h_bot_, w_bot_;
        int n_top_, c_top_, h_top_, w_top_;
        int ph_, pw_, sh_, sw_, kh_, kw_;
        int size_bot_, size_top_;
        size_t size_g_;
        cl_ulong s_time, e_time;
        cl_event event_maxpooling;
};
#endif

#ifdef INNER_PRODUCT
class InnerProductLayer {
    public:
        InnerProductLayer();
        ~InnerProductLayer();
        InnerProductLayer(int dn, int dc, int dh, int dw, int fn, float *weight, float *bias);
        InnerProductLayer(int flag, int dn, int dc, int dh, int dw, int fn, float *weight, float *bias);
        void get_mem(int &dn, int &dc, int &dh, int &dw);
        void release_mem(void);
        void forward(cl_mem data);
        cl_mem inner_;
        double c_time;
    private:
        int col_;
        size_t row_;
        cl_mem weight_, bias_;
        cl_ulong s_time, e_time;
        cl_event event_innerproduct;
};
#endif

#ifdef SIGMOID
class SigmoidLayer {
    public:
        SigmoidLayer();
        ~SigmoidLayer();
        SigmoidLayer(int dn, int dc, int dh, int dw);
        void forward(cl_mem &data);
        double c_time;
    private:
        size_t col_;
        cl_ulong s_time, e_time;
        cl_event event_sigmoid;
};
#endif

#endif

