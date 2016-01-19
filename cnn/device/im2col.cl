/*
* im2col
* stride w = stride h = 1 
* filter w = filter h
* pad w = pad h = filter h / 2
* resource:
* DSP block: 
*/
#ifdef TASK_IM2COL
kernel __attribute__((task))
void im2col (__global float *restrict data_img,
             __global float *restrict data_col,
             const int offset_img, const int offset_col,
             const int height_img, const int width_img,
             const int height_col, const int width_col,
             const int size_kernel, const int size_pad,
             const int w_offset, const int h_offset,
             const int size_stride) {
    int w_pad, h_pad;
    int index_img, index_col;
    
    __global float * restrict ptr_col = data_col + offset_col;
    __global float * restrict ptr_img = data_img + offset_img;

    for(int h = 0; h < height_col; h++) {
        for(int w = 0; w < width_col; w++) {
            w_pad = w * size_stride - size_pad + w_offset;
            h_pad = h * size_stride - size_pad + h_offset;
            index_img = h_pad * width_img + w_pad;
            index_col = h * width_col + w;
            if(w_pad >= 0 && w_pad < width_img && h_pad >= 0 && h_pad < height_img)
                ptr_col[index_col] = ptr_img[index_img];
            else
                ptr_col[index_col] = 0;
        }    
    }
}
#elif defined(EXT_IM2COL)
__kernel
__attribute((reqd_work_group_size(BS_IM2COL, 1, 1)))
void im2col (__global float *restrict data_img,
             __global float *restrict data_col,
             const int offset_img, const int offset_col,
             const int height_img, const int width_img,
             const int height_col, const int width_col,
             const int size_kernel, const int size_pad, 
             const int size_stride) {
    int index       = get_global_id(0);
    int temp        = index / size_kernel;
    int w_offset    = index - temp * size_kernel;
    int c_offset    = temp / size_kernel;
    int h_offset    = temp - c_offset * size_kernel;
    int w_pad, h_pad;
    __global float * restrict ptr_col = data_col;
    ptr_col += index * offset_col;
    __global float * restrict ptr_img = data_img;
    ptr_img += c_offset * offset_img;

    for(int h = 0; h < height_col; h++) {
        for(int w = 0; w < width_col; w++) {
            w_pad = w * size_stride - size_pad + w_offset;
            h_pad = h * size_stride - size_pad + h_offset;
            if(w_pad >= 0 && w_pad < width_img && h_pad >= 0 && h_pad < height_img)
                ptr_col[h*width_col+w] = ptr_img[h_pad*width_img+w_pad];
            else
                ptr_col[h*width_col+w] = 0;
        } 
    }
}
#else
__kernel
__attribute((reqd_work_group_size(BS, 1, 1)))
void im2col (__global const float *restrict data_img,
             __global float *restrict data_col,
             const uint offset_img, const uint offset_col
             const uint height_img, const uint width_img,
             const uint height_col, const uint width_col,
             const uint size_kernel, const uint size_pad) {
    int index       = get_global_id(0);
    int w_out       = index % width_col;
    int h_index     = index / width_col;
    int h_out       = h_index % height_col;
    int channel_in  = h_index / height_col;
    int channel_out = channel_in * size_kernel * size_kernel;
    int h_in        = h_out - size_pad;
    int w_in        = w_out - size_pad;
    __global float * data_col_ptr = data_col;
    data_col_ptr   += (channel_out * height_col + h_out) * width_col + w_out;
    __global const float * data_im_ptr = data_im;
    data_im_ptr    += (channel_in * height_im + h_in) * width_im + w_in;

    for(int i = 0; i < size_kernel; i++){
        for(int j = 0; j < size_kernel; j++) {
            int h = h_in + i;
            int w = w_in + j;
            if(h >= 0 && w >= 0 && h < height_im && w < width_im)
                *data_col_ptr = data_im_ptr[i * width + j];
            else
                *data_col_ptr = 0;
            data_col_ptr += height_col * width_col;
        }    
    }
}
#endif
