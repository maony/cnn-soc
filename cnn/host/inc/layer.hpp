#ifndef __LAYER__
#define __LAYER__
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#include "common.hpp"

using namespace aocl_utils;

extern cl_platform_id platform;
extern cl_device_id device;
extern cl_context context;
extern cl_command_queue queues[K_NUM_KERNELS];
extern cl_kernel kernels[K_NUM_KERNELS];
extern cl_program program;
extern cl_int status;

class ConvLayer {
    public:
        ConvLayer(void);
        ~ConvLayer();
        ConvLayer(int dn, int dc, int dh, int dw, int ph, int pw, int sh, int sw);
        void init_weight(int fn, int fh, int fw, float *weight, float *bias);
        void get_mem(cl_mem &data, int &dn, int &dc, int &dh, int &dw);
        void forward(cl_mem bot);
    private:
        int n_bot_, c_bot_, h_bot_, w_bot_;
        int n_top_, c_top_, h_top_, w_top_;
        int ph_, pw_, sh_, sw_;
        int fn_, fc_, fh_, fw_;
        int offset_bot_, offset_top_;
        cl_mem weight_, bias_, top_, col_;
};

class PreluLayer {
    public:
        PreluLayer();
        PreluLayer(int dn, int dc, int dh, int dw, float *slope);
        ~PreluLayer();
    private:
        int n_, c_, h_, w_;
        cl_mem slope_;
}

#endif

