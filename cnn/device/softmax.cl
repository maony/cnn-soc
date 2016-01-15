/*
* softmax 
* stride w = stride h = 1 
* filter w = filter h
* pad w = pad h = filter h / 2
* resource:
* DSP block: 
*/
#define BS_SOFTMAX 1

__kernel
__attribute((reqd_work_group_size(BS_SOFTMAX, 1, 1)))
void softmax (__global float *restrict data_input,
              __global float *restrict data_output,
              const int num) {
    float scale = data_input[0], value = 0, sum = 0;
    int i;
    
    for(i = 1; i < num; i++) {
        value = data_input[i];
        scale = (scale > value) ? scale : value;
    }

    for(i = 0; i < num; i++) {
        value = data_input[i] - scale;
        value = (float)exp(value);
        sum  += value;
        data_out[i] = value;
    }

    for(i = 0; i < num; i++) {
        data_out[i] = data_out[i] / sum;    
    }
}
