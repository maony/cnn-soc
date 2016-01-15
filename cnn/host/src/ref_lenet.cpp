#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "ref_cnn.hpp"



float   src_col[96 * 25 * 27 * 27];

void load_model_param(FILE *fp, uint offset, uint num, float *param)
{

	fseek(fp, sizeof(float)*offset, 0);
	fread(param, sizeof(float), num, fp);
}

int sDiv(int num, int den)
{
	return num / den;
}

void im2col(const Dtype* data_im, Dtype* data_col, const int channels,
	const int height, const int width, const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w,
	const int stride_h, const int stride_w)
{
	/*int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
	int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
	int channels_col = channels * kernel_h * kernel_w;
	int c, w_offset, h_offset, c_im, h, w, h_pad, w_pad;

	for (c = 0; c < channels_col; ++c) {
		w_offset = c - (c / kernel_w) * kernel_w;
		h_offset = (c / kernel_w) - ((c / kernel_w) / kernel_h) * kernel_h;
		c_im = ((c / kernel_w) / kernel_h);

		for (h = 0; h < height_col; ++h) {
			for (w = 0; w < width_col; w += 1) {
				h_pad = h * stride_h - pad_h + h_offset;
				w_pad = w * stride_w - pad_w + w_offset;

				if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
					data_col[(c * height_col + h) * width_col + w] =
					data_im[(c_im * height + h_pad) * width + w_pad];
				else
					data_col[(c * height_col + h) * width_col + w] = 0;
			}
		}
	}*/

	int height_col = sDiv((height + 2 * pad_h - kernel_h), stride_h) + 1;
	int width_col = sDiv((width + 2 * pad_w - kernel_w), stride_w) + 1;
	int channels_col = channels * kernel_h * kernel_w;
	int c, w_offset, h_offset, c_im, h, w, h_pad, w_pad;

	int h_start, h_end, w_start, w_end;

	for (c = 0; c < channels_col; ++c) {
		w_offset = c - sDiv(c, kernel_w) * kernel_w;
		h_offset = sDiv(c, kernel_w) - sDiv(sDiv(c, kernel_w), kernel_h) * kernel_h;
		c_im = sDiv(sDiv(c, kernel_w), kernel_h);

		h_start = (pad_h - h_offset) / stride_h;
		h_end = (height + pad_h - h_offset) / stride_h;
		w_start = (pad_w - w_offset) / stride_w;
		w_end = (width + pad_w - w_offset) / stride_w;

		h_start = MAX(h_start, 0);
		h_end = MIN(h_end, height_col);
		w_start = MAX(w_start, 0);
		w_end = MIN(w_end, width_col);

		for (h = h_start; h < h_end; ++h) {
			for (w = w_start; w < w_end; w += 1) {
				h_pad = h * stride_h - pad_h + h_offset;
				w_pad = w * stride_w - pad_w + w_offset;

				data_col[(c * height_col + h) * width_col + w] =
					data_im[(c_im * height + h_pad) * width + w_pad];
			}
		}

		for (h = 0; h < h_start; ++h)
			for (w = 0; w < width_col; w += 1)
				data_col[(c * height_col + h) * width_col + w] = 0;
		for (h = h_end; h < height_col; ++h)
			for (w = 0; w < width_col; w += 1)
				data_col[(c * height_col + h) * width_col + w] = 0;

		for (h = h_start; h < h_end; ++h) {
			for (w = 0; w < w_start; w += 1)
				data_col[(c * height_col + h) * width_col + w] = 0;
			for (w = w_end; w < width_col; w += 1)
				data_col[(c * height_col + h) * width_col + w] = 0;
		}
	}

}

/* Native implementation for reference */
void MatMul(float * NO_ALIAS a, float * NO_ALIAS b, float * NO_ALIAS c, const int m, const int k, const int n)
{
	int ii, jj, kk;
	float temp = 0;

	for (ii = 0; ii < m; ii++)
		for (jj = 0; jj < n; jj++)
		{
			temp = 0.0;
			for (kk = 0; kk < k; kk++)
			{
				temp += a[ii*k + kk] * b[n*kk + jj];
			}
			*(c + ii*n + jj) = temp;
		}
}

void conv_layer(const blob_shape_t src, const blob_shape_t filter, blob_shape_t dst, const conv_param_t param)
{

	float * NO_ALIAS img_col = src_col;
	float * NO_ALIAS img_dst = dst.data;

	ASSERT(src.channels == filter.channels);

	// image to col
	im2col(src.data, img_col, src.channels,
		src.height, src.width, filter.height, filter.width,
		param.pad_h, param.pad_w, param.stride_h, param.stride_w);
	// Matrix mul
	MatMul(filter.data, img_col, img_dst, filter.num, filter.channels*filter.width*filter.height, dst.width*dst.height);
}

void conv_bias(float * NO_ALIAS data, uint row, uint col, float * NO_ALIAS bias)
{
	int r, c;
	for (r = 0; r < row; r++)
		for (c = 0; c < col; c++)
			*(data + r*col + c) = (*(data + r*col + c)) + (*(bias + r));
}

void pooling_max_layer(const blob_shape_t src, const conv_param_t param, const int kernel_h, const int kernel_w, blob_shape_t dst)
{
	int n, c, ph, pw, kh, kw;
	int hstart, hend, wstart, wend;
	int pindex, kindex;
	float *psrc, *pdst;

	psrc = src.data;
	pdst = dst.data;

	for (n = 0; n < dst.num; n++){
		for (c = 0; c < dst.channels; c++){
			for (ph = 0; ph < dst.height; ph++){
				for (pw = 0; pw < dst.width; pw++){
					pindex = ((n*dst.channels + c)*dst.height + ph)*dst.width + pw;
					pdst[pindex] = -FLT_MAX;
				}
			}
		}
	}

	for (n = 0; n < dst.num; n++){
		for (c = 0; c < dst.channels; c++){
			for (ph = 0; ph < dst.height; ph++){
				for (pw = 0; pw < dst.width; pw++){
					hstart = ph * param.stride_h - param.pad_h;
					wstart = pw * param.stride_w - param.pad_w;
					hend = MIN(hstart + kernel_h, src.height);
					wend = MIN(wstart + kernel_w, src.width);
					hstart = MAX(hstart, 0);
					wstart = MAX(wstart, 0);
					pindex = ph * dst.width + pw;
					for (kh = hstart; kh < hend; kh++){
						for (kw = wstart; kw < wend; kw++){
							kindex = kh * src.width + kw;
							if (psrc[kindex] > pdst[pindex])
								pdst[pindex] = psrc[kindex];
						}
					}
				}
			}
			psrc += src.width * src.height;
			pdst += dst.width * dst.height;
		}
	}
}

// Matrix X vector
void inner_product_layer(float * NO_ALIAS in_data, float * NO_ALIAS in_weight, float * NO_ALIAS in_bias, float * NO_ALIAS out_data, uint row, uint col)
{
	int r, c;
	float temp;

	for (r = 0; r < row; r++){
		temp = 0;
		for (c = 0; c < col; c++){
			temp += (*(in_data + c) * (*(in_weight + r*col + c)));
		}
		*(out_data + r) = temp + (*(in_bias + r));
	}
}

// Relu layer
void relu_layer(float * NO_ALIAS in_data, float * NO_ALIAS out_data, uint num)
{
	int n;
	for (n = 0; n < num; n++)
		out_data[n] = (in_data[n] < 0) ? (float)0 : in_data[n];
}

void softmax_layer(float * NO_ALIAS in_data, float * NO_ALIAS out_data, uint num)
{
	int n;
	float scale = in_data[0];
	// Max
	for (n = 1; n < num; n++)
		scale = MAX(scale, in_data[n]);
	// Subtract max
	for (n = 0; n < num; n++)
		out_data[n] = in_data[n] - scale;
	// exp
	for (n = 0; n < num; n++)
		out_data[n] = (float)exp(out_data[n]);
	// sum
	scale = 0.0;
	for (n = 0; n < num; n++)
		scale += out_data[n];
	// division
	for (n = 0; n < num; n++)
		out_data[n] /= scale;
}

void lenet(uchar *img, uchar *model_file, int file_sel)
{
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

	//fopen_s(&fp_data, img, "rb");
	//fopen_s(&fp_model, model_file, "rb");
	fp_data     = fopen(img, "rb");
	fp_model    = fopen(model_file, "rb");
	
	// Convolution layer 1
	// num:20, kernel:5, stride:1
	// parameter, weight: 5*5*20; bias:20
	conv1_data.num = 1;
	conv1_data.channels = 1;
	conv1_data.height = 28;
	conv1_data.width = 28;
	conv1_data.data = (float *)malloc(conv1_data.num*conv1_data.channels*conv1_data.height*conv1_data.width*sizeof(float));
	// =============================================================
	// TODO: load data in
	fseek(fp_data, sizeof(float)*file_sel*28*28, 0);
	fread(conv1_data.data, sizeof(float), conv1_data.num*conv1_data.channels*conv1_data.height*conv1_data.width, fp_data);
	// =============================================================

	conv1_filter.num = 20;
	conv1_filter.channels = 1;
	conv1_filter.height = 5;
	conv1_filter.width = 5;
	conv1_filter.data = (float *)malloc(conv1_filter.num*conv1_filter.channels*conv1_filter.height*conv1_filter.width*sizeof(float));
	conv1_param.pad_w = 0;
	conv1_param.pad_h = 0;
	conv1_param.stride_w = 1;
	conv1_param.stride_h = 1;
	conv1_out.num = 1;
	conv1_out.channels = 20;
	conv1_out.height = (conv1_data.height + 2 * conv1_param.pad_h - conv1_filter.height) / conv1_param.stride_h + 1;
	conv1_out.width = (conv1_data.width + 2 * conv1_param.pad_w - conv1_filter.width) / conv1_param.stride_w + 1;
	conv1_out.data = (float *)malloc(conv1_out.num*conv1_out.channels*conv1_out.height*conv1_out.width*sizeof(float));

	conv1_bias = (float *)malloc(20 * sizeof(float));

	load_model_param(fp_model, offset, 5 * 5 * 20, conv1_filter.data);
	offset += 5 * 5 * 20;
	load_model_param(fp_model, offset, 20, conv1_bias);
	offset += 20;
	// 20X(5*5), 25X(24*24)
	conv_layer(conv1_data, conv1_filter, conv1_out, conv1_param);
	conv_bias(conv1_out.data, conv1_out.channels, conv1_out.height*conv1_out.width, conv1_bias);

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
	pool1_out.data = (float *)malloc(pool1_out.num*pool1_out.channels*pool1_out.height*pool1_out.width*sizeof(float));

	pooling_max_layer(conv1_out, pool1_param, pool1_param.stride_h, pool1_param.stride_w, pool1_out);

	// Convolution layer 2
	// num:50, kernel:5, stride:1
	// parameter, weight: 5*5*20; bias:50
	conv2_filter.num = 50+2;
	conv2_filter.channels = 20;
	conv2_filter.height = 5;
	conv2_filter.width = 5;
	conv2_filter.data = (float *)malloc((conv2_filter.num)*conv2_filter.channels*conv2_filter.height*conv2_filter.width*sizeof(float));
	conv2_param.pad_w = 0;
	conv2_param.pad_h = 0;
	conv2_param.stride_w = 1;
	conv2_param.stride_h = 1;
	conv2_out.num = 1;
	conv2_out.channels = 50;
	conv2_out.height = (pool1_out.height + 2 * conv2_param.pad_h - conv2_filter.height) / conv2_param.stride_h + 1;
	conv2_out.width = (pool1_out.width + 2 * conv2_param.pad_w - conv2_filter.width) / conv2_param.stride_w + 1;
	conv2_out.data = (float *)malloc(conv2_out.num*(conv2_out.channels+2)*conv2_out.height*conv2_out.width*sizeof(float));

	conv2_bias = (float *)malloc(50 * sizeof(float));

	load_model_param(fp_model, offset, 5 * 5 * 20 * 50, conv2_filter.data);
	offset += 5 * 5 * 20 * 50;
	load_model_param(fp_model, offset, 50, conv2_bias);
	offset += 50;
	conv_layer(pool1_out, conv2_filter, conv2_out, conv2_param);
	conv_bias(conv2_out.data, conv2_out.channels, conv2_out.height*conv2_out.width, conv2_bias);

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
	pool2_out.data = (float *)malloc(pool2_out.num*pool2_out.channels*pool2_out.height*pool2_out.width*sizeof(float));
	pooling_max_layer(conv2_out, pool2_param, pool2_param.stride_h, pool2_param.stride_w, pool2_out);

	// Inner-product layer 1
	// num: 500
	inner1_out = (float *)malloc(500 * 1 * sizeof(float));
	inner1_weight = (float *)malloc(500 * 50 * pool2_out.height*pool2_out.width*sizeof(float));
	inner1_bias = (float *)malloc(500 * sizeof(float));
	load_model_param(fp_model, offset, 500 * 50 * pool2_out.height*pool2_out.width, inner1_weight);
	offset += 500 * 50 * pool2_out.height*pool2_out.width;
	load_model_param(fp_model, offset, 500, inner1_bias);
	offset += 500;
	inner_product_layer(pool2_out.data, inner1_weight, inner1_bias, inner1_out, 500, 50 * pool2_out.height*pool2_out.width);

	// ReLu layer 1
	relu1_out = (float *)malloc(500 * sizeof(float));
	relu_layer(inner1_out, relu1_out, 500);

	// Inner-product layer 2
	// num: 10
	inner2_out = (float *)malloc(10 * 1 * sizeof(float));
	inner2_weight = (float *)malloc(10 * 500 * sizeof(float));
	inner2_bias = (float *)malloc(10 * 1 * sizeof(float));
	load_model_param(fp_model, offset, 10 * 500, inner2_weight);
	offset += 10 * 500;
	load_model_param(fp_model, offset, 10, inner2_bias);
	offset += 10;
	inner_product_layer(relu1_out, inner2_weight, inner2_bias, inner2_out, 10, 500);

	// Softmax layer 1
	soft1_out = (float *)malloc(10 * 1 * sizeof(float));
	softmax_layer(inner2_out, soft1_out, 10);

	//printf("inner2_out output value:\n");
	//for (offset = 0; offset < 10; offset++)
	//	printf("%.8f  ", inner2_out[offset]);

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

void ref_lenet(void)
{
	uchar img[] = "data-all.bin";
	uchar model[] =  "model.bin";

	int sel;
	for (sel = 0; sel < 10; sel++)
	{
		printf("test data is %d:\n", sel);
        double time_cost = getCurrentTimestamp();
		lenet(img, model, sel);
        printf("time cost is %5.2lf\n", (getCurrentTimestamp() - time_cost) * 1e-6);
	}
		
}
