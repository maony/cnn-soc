
#include "layer.hpp"


#if defined(IM2COL) && defined(GEMM)
ConvLayer::ConvLayer(){}

ConvLayer::ConvLayer(int dn, int dc, int dh, int dw, int ph, int pw, int sh, int sw) {
    n_bot_ = dn;    c_bot_ = dc;
    h_bot_ = dh;    w_bot_ = dw;
    ph_    = ph;    pw_    = pw;
    sh_    = sh;    sw_    = sw;
}

ConvLayer::~ConvLayer() {
    if(top_)        clReleaseMemObject(top_);    
    if(weight_)     clReleaseMemObject(weight_);    
    if(bias_)       clReleaseMemObject(bias_);
    if(col_)        clReleaseMemObject(col_);    
}

//void ConvLayer::init_param(int dn, int dc, int dh, int dw, int ph, int pw, int sh, int sw) {
//    n_bot_ = dn;    c_bot_ = dc;
//    h_bot_ = dh;    w_bot_ = dw;
//    ph_    = ph;    pw_    = pw;
//    sh_    = sh;    sw_    = sw;
//}

void ConvLayer::init_weight(int fn, int fh, int fw, float *weight, float *bias) {
    fn_     = fn;   fc_     = c_bot_;
    fh_     = fh;   fw_     = fw;
    n_top_  = 1;    c_top_  = fn_;
    h_top_  = (h_bot_ + 2 * ph_ - fh_) / sh_ + 1;
    w_top_  = (w_bot_ + 2 * pw_ - fw_) / sw_ + 1;
    offset_bot_ = h_bot_ * w_bot_;
    offset_top_ = h_top_ * w_top_;
    int M = fn_, K = fc_ * fh_ * fw_, N = h_top_ * w_top_;

    weight_  = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * M * K, NULL, &status);
    checkError(status, "Failed to allocate col buffer\n");
    status = clEnqueueWriteBuffer(queues[K_GEMM], weight_, CL_TRUE, 0, sizeof(float) * M * K, weight, 0, NULL, NULL);
    checkError(status, "Failed to copy data to device");
    bias_    = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * c_top_, NULL, &status);
    checkError(status, "Failed to allocate bias buffer\n");
    status = clEnqueueWriteBuffer(queues[K_GEMM], bias_, CL_TRUE, 0, sizeof(float) * c_top_, bias, 0, NULL, NULL);
    checkError(status, "Failed to copy data to device");
    clFinish(queues[K_GEMM]);
    //clFinish(queues[K_BIAS]);
   
    //printf(" data is %e %e\n", *bias, *(bias+1));
    //printf(" data is %e %e\n", *weight, *(weight+1));
    //float *h_out = (float *)alignedMalloc(sizeof(float) * n_top_ * c_top_ * h_top_ * w_top_);
    //status = clEnqueueReadBuffer(queues[K_IM2COL], col_, CL_TRUE, 0, sizeof(float) * n_top_ * c_top_ * h_top_ * w_top_, h_out, 0, NULL, NULL);
    //clFinish(queues[K_IM2COL]);
    //printf(" data is %e %e\n", *h_out, *(h_out+1));

    col_     = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * c_bot_ * h_top_ * w_top_ * fh_ * fw_, NULL, &status);
    checkError(status, "Failed to allocate col buffer\n");
    
    top_     = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * M * N, NULL, &status);
    checkError(status, "Failed to allocate col buffer\n");
}

void ConvLayer::forward(cl_mem bot) {
    status = clSetKernelArg(kernels[K_IM2COL], 0, sizeof(cl_mem), (void*)&bot);
    checkError(status, "Failed to set im2col arg 0");
    status = clSetKernelArg(kernels[K_IM2COL], 1, sizeof(cl_mem), (void*)&col_);
    checkError(status, "Failed to set im2col arg 1");
    status = clSetKernelArg(kernels[K_IM2COL], 2, sizeof(int), (void*)&offset_bot_);
    checkError(status, "Failed to set im2col arg 2");
    status = clSetKernelArg(kernels[K_IM2COL], 3, sizeof(int), (void*)&offset_top_);
    checkError(status, "Failed to set im2col arg 3");
    status = clSetKernelArg(kernels[K_IM2COL], 4, sizeof(int), (void*)&h_bot_);
    checkError(status, "Failed to set im2col arg 4");
    status = clSetKernelArg(kernels[K_IM2COL], 5, sizeof(int), (void*)&w_bot_);
    checkError(status, "Failed to set im2col arg 5");
    status = clSetKernelArg(kernels[K_IM2COL], 6, sizeof(int), (void*)&h_top_);
    checkError(status, "Failed to set im2col arg 6");
    status = clSetKernelArg(kernels[K_IM2COL], 7, sizeof(int), (void*)&w_top_);
    checkError(status, "Failed to set im2col arg 7");
    status = clSetKernelArg(kernels[K_IM2COL], 8, sizeof(int), (void*)&fh_);
    checkError(status, "Failed to set im2col arg 8");
    status = clSetKernelArg(kernels[K_IM2COL], 9, sizeof(int), (void*)&ph_);
    checkError(status, "Failed to set im2col arg 9");
    status = clSetKernelArg(kernels[K_IM2COL], 10, sizeof(int), (void*)&sh_);
    checkError(status, "Failed to set im2col arg 10");
    
    size_t channel_ext = c_bot_ * fh_ * fw_;
    status = clEnqueueNDRangeKernel(queues[K_IM2COL], kernels[K_IM2COL], 1, NULL, &channel_ext, NULL, 0, NULL, NULL);
    
    //float *h_out = (float *)alignedMalloc(sizeof(float) * n_top_ * c_top_ * h_top_ * w_top_);
    //status = clEnqueueReadBuffer(queues[K_IM2COL], col_, CL_TRUE, 0, sizeof(float) * n_top_ * c_top_ * h_top_ * w_top_, h_out, 0, NULL, NULL);
    //clFinish(queues[K_IM2COL]);
    //printf(" datad sd is %e %e\n", *h_out, *(h_out+128 + 128 + 8));
    
    int M = fn_, K = fc_ * fh_ * fw_, N = h_top_ * w_top_;
    size_t wg_size[2]   = {1, 1};
    size_t g_size[2]    = {N, M};
    clFinish(queues[K_IM2COL]);
    
    status     = clSetKernelArg(kernels[K_GEMM], 0, sizeof(cl_mem), &top_);
    status    |= clSetKernelArg(kernels[K_GEMM], 1, sizeof(cl_mem), &weight_);
    status    |= clSetKernelArg(kernels[K_GEMM], 2, sizeof(cl_mem), &col_);
    status    |= clSetKernelArg(kernels[K_GEMM], 3, sizeof(int), &M);
    status    |= clSetKernelArg(kernels[K_GEMM], 4, sizeof(int), &K);
    status    |= clSetKernelArg(kernels[K_GEMM], 5, sizeof(int), &N);
    status    |= clSetKernelArg(kernels[K_GEMM], 6, sizeof(cl_mem), &bias_);
    status = clEnqueueNDRangeKernel(queues[K_GEMM], kernels[K_GEMM], 2, NULL, g_size, wg_size, 0, NULL, NULL);
    clFinish(queues[K_GEMM]);
    
    //status     = clSetKernelArg(kernels[K_BIAS], 0, sizeof(cl_mem), &top_);
    //status    |= clSetKernelArg(kernels[K_BIAS], 1, sizeof(cl_mem), &bias_);
    //status    |= clSetKernelArg(kernels[K_BIAS], 2, sizeof(int), &N);
    //g_size[0] = c_top_;
    //status = clEnqueueNDRangeKernel(queues[K_BIAS], kernels[K_BIAS], 1, NULL, &g_size[0], NULL, 0, NULL, NULL);
    //clFinish(queues[K_BIAS]);
    
    //float *h_out = (float *)alignedMalloc(sizeof(float) * n_top_ * c_top_ * h_top_ * w_top_);
    //status = clEnqueueReadBuffer(queues[K_IM2COL], col_, CL_TRUE, 0, sizeof(float) * n_top_ * c_top_ * h_top_ * w_top_, h_out, 0, NULL, NULL);
    //clFinish(queues[K_IM2COL]);
    //printf(" data is %e %e\n", *h_out, *(h_out+1));
}

void ConvLayer::get_mem(int &n, int &c, int &h, int &w) {
    n = n_top_;
    c = c_top_;
    h = h_top_;
    w = w_top_;
}
#endif


#ifdef PRELU
PreluLayer::PreluLayer() {}
PreluLayer::~PreluLayer() {
    if(slope_)      clReleaseMemObject(slope_);
}
PreluLayer::PreluLayer(int dn, int dc, int dh, int dw, float* slope) {
    n_ = dn;
    c_ = dc;
    h_ = dh;
    w_ = dw;
    size_ = h_ * w_;
    g_size_ = c_;

    slope_  = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * c_, NULL, &status);
    checkError(status, "Failed to allocate col buffer\n");
    status = clEnqueueWriteBuffer(queues[K_PRELU], slope_, CL_TRUE, 0, sizeof(float) * c_, slope, 0, NULL, NULL);
    checkError(status, "Failed to copy data to device");
    clFinish(queues[K_PRELU]);
}

void PreluLayer::forward(cl_mem &data) {
    status     = clSetKernelArg(kernels[K_PRELU], 0, sizeof(cl_mem), &data);
    status    |= clSetKernelArg(kernels[K_PRELU], 1, sizeof(cl_mem), &slope_);
    status    |= clSetKernelArg(kernels[K_PRELU], 2, sizeof(int), &size_);
    
    status = clEnqueueNDRangeKernel(queues[K_PRELU], kernels[K_PRELU], 1, NULL, &g_size_, NULL, 0, NULL, NULL);
    clFinish(queues[K_PRELU]);
}
#endif

#ifdef MAX_POOLING
MaxPoolingLayer::MaxPoolingLayer() {}
MaxPoolingLayer::~MaxPoolingLayer() {
    if(pool_)   clReleaseMemObject(pool_);
}
MaxPoolingLayer::MaxPoolingLayer(int dn, int dc, int dh, int dw, int ph, int pw, int sh, int sw, int kh, int kw) {
    n_bot_ = dn;    c_bot_ = dc;    h_bot_ = dh;    w_bot_ = dw;
    ph_    = ph;    pw_    = pw;    sh_    = sh;    sw_    = sw;
    kh_    = kh;    kw_    = kw;
    n_top_ = n_bot_;    c_top_ = c_bot_;
    h_top_ = (h_bot_ + 2 * ph_ - sh_) / sh_ + 1;
    w_top_ = (w_bot_ + 2 * pw_ - sw_) / sw_ + 1;
    size_bot_ = h_bot_ * w_bot_;
    size_top_ = h_top_ * w_top_;
    size_g_   = c_top_ * size_top_;

    pool_  = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * n_top_ * c_top_ * size_top_, NULL, &status);
    checkError(status, "Failed to allocate pool buffer\n");
}

void MaxPoolingLayer::get_mem(int &n, int &c, int &h, int &w) {
    n = n_top_;
    c = c_top_;
    h = h_top_;
    w = w_top_;
}

void MaxPoolingLayer::forward(cl_mem data) {
    status     = clSetKernelArg(kernels[K_MAX_POOLING], 0, sizeof(cl_mem), &data);
    status    |= clSetKernelArg(kernels[K_MAX_POOLING], 1, sizeof(cl_mem), &pool_);
    status    |= clSetKernelArg(kernels[K_MAX_POOLING], 2, sizeof(int), &h_bot_);
    status    |= clSetKernelArg(kernels[K_MAX_POOLING], 3, sizeof(int), &w_bot_);
    status    |= clSetKernelArg(kernels[K_MAX_POOLING], 4, sizeof(int), &h_top_);
    status    |= clSetKernelArg(kernels[K_MAX_POOLING], 5, sizeof(int), &w_top_);
    status    |= clSetKernelArg(kernels[K_MAX_POOLING], 6, sizeof(int), &size_bot_);
    status    |= clSetKernelArg(kernels[K_MAX_POOLING], 7, sizeof(int), &sh_);
    status    |= clSetKernelArg(kernels[K_MAX_POOLING], 8, sizeof(int), &kh_);
    status    |= clSetKernelArg(kernels[K_MAX_POOLING], 9, sizeof(int), &ph_);
    
    status = clEnqueueNDRangeKernel(queues[K_MAX_POOLING], kernels[K_MAX_POOLING], 1, NULL, &size_g_, NULL, 0, NULL, NULL);
    clFinish(queues[K_MAX_POOLING]);
}
#endif

#ifdef INNER_PRODUCT
InnerProductLayer::InnerProductLayer() {}
InnerProductLayer::~InnerProductLayer() {
    if(inner_)      clReleaseMemObject(inner_);
    if(weight_)     clReleaseMemObject(weight_);
    if(bias_)       clReleaseMemObject(bias_);
}
InnerProductLayer::InnerProductLayer(int dn, int dc, int dh, int dw, int fn, float *weight, float *bias) {
    row_ = fn;
    col_ = dn * dc * dh * dw;
    printf("row %d, col %d\n", row_, col_);

    inner_  = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * row_, NULL, &status);
    checkError(status, "Failed to allocate inner buffer\n");
    //if(status != CL_SUCCESS) printf("create inner error\n");
    weight_ = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * row_ * col_, NULL, &status);
    checkError(status, "Failed to allocate inner buffer\n");
    bias_   = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * row_, NULL, &status);
    checkError(status, "Failed to allocate inner buffer\n");
    
    status = clEnqueueWriteBuffer(queues[K_PRELU], weight_, CL_TRUE, 0, sizeof(float) * row_ * col_, weight, 0, NULL, NULL);
    checkError(status, "Failed to copy data to device");
    clFinish(queues[K_PRELU]);
    status = clEnqueueWriteBuffer(queues[K_INNER_PRODUCT], bias_, CL_TRUE, 0, sizeof(float) * row_, bias, 0, NULL, NULL);
    checkError(status, "Failed to copy data to device");
    clFinish(queues[K_INNER_PRODUCT]);
}

void InnerProductLayer::forward(cl_mem data) {
    status     = clSetKernelArg(kernels[K_INNER_PRODUCT], 0, sizeof(cl_mem), &data);
    checkError(status, "Failed to set arg 0");
    status    |= clSetKernelArg(kernels[K_INNER_PRODUCT], 1, sizeof(cl_mem), &weight_);
    checkError(status, "Failed to set arg 1");
    status    |= clSetKernelArg(kernels[K_INNER_PRODUCT], 2, sizeof(cl_mem), &bias_);
    checkError(status, "Failed to set arg 2");
    status    |= clSetKernelArg(kernels[K_INNER_PRODUCT], 3, sizeof(cl_mem), &inner_);
    checkError(status, "Failed to set arg 3");
    status    |= clSetKernelArg(kernels[K_INNER_PRODUCT], 4, sizeof(int), &col_);
    checkError(status, "Failed to set arg");
    
    printf("start inner kernel\n");
    status = clEnqueueNDRangeKernel(queues[K_INNER_PRODUCT], kernels[K_INNER_PRODUCT], 1, NULL, &row_, NULL, 0, NULL, NULL);
    checkError(status, "Failed to exec");
    printf("end inner kernel\n");
    clFinish(queues[K_INNER_PRODUCT]);
}

void InnerProductLayer::get_mem(int &dn, int &dc, int &dh, int &dw) {
    dn = 1;
    dc = row_;
    dh = 1;
    dw = 1;
}
#endif

