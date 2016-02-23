/*
* max_pooling
* stride w = stride h = 1 
* filter w = filter h
* pad w = pad h = filter h / 2
* resource:
* DSP block: 
*/
#define BS_MAX_POOLING 1

__kernel
__attribute((reqd_work_group_size(BS_MAX_POOLING, 1, 1)))
void max_pooling (__global float *restrict data_input,
                  __global float *restrict data_output,
             const int height, const int width,
             const int height_pool, const int width_pool,
             const int offset_input, const int size_stride,
             const int size_kernel, const int size_pad) {
    int index       = get_global_id(0);

    int temp        = index / width_pool;
    int pw          = index - temp * width_pool;
    int c           = temp / height_pool;
    int ph          = temp - c * height_pool;
    int hstart      = ph * size_stride - size_pad;
    int wstart      = pw * size_stride - size_pad;
    int hend        = ((hstart + size_kernel) < height) ? (hstart + size_kernel) : height;
    int wend        = ((wstart + size_kernel) < width) ? (wstart + size_kernel) : width;
    hstart          = (hstart > 0) ? hstart : 0;
    wstart          = (wstart > 0) ? wstart : 0;
    float maxval    = -FLT_MAX;
    __global float *restrict ptr_input = data_input + c * offset_input; 
    int offset;
    float iter;

    #pragma unroll 0
    for(int h = hstart; h < hend; h++) {
        #pragma unroll 0
        for(int w = wstart; w < wend; w++) {
            offset = h * width + w;
            iter = ptr_input[offset];
            maxval = (iter > maxval) ? iter : maxval;
        }    
    }

    data_output[index] = maxval;
}
