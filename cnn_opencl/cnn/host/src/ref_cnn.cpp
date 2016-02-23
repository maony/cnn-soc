#include "ref_cnn.hpp"

/* Native implementation for reference */
void im2col_ref(const float* data_im, float* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w) 
{
    int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    int channels_col = channels * kernel_h * kernel_w;
    int c, w_offset, h_offset, c_im, h, w, h_pad, w_pad;
    
    for (c = 0; c < channels_col; ++c) {
        w_offset = c % kernel_w;
        h_offset = (c / kernel_w) % kernel_h;
        c_im = c / kernel_h / kernel_w;
    
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                h_pad = h * stride_h - pad_h + h_offset;
                w_pad = w * stride_w - pad_w + w_offset;
                if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
                  data_col[(c * height_col + h) * width_col + w] =
                    data_im[(c_im * height + h_pad) * width + w_pad];
                else
                  data_col[(c * height_col + h) * width_col + w] = 0;
            }
        }
    }
}
