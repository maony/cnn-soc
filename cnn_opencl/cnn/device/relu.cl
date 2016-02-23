/*
* relu 
* stride w = stride h = 1 
* filter w = filter h
* pad w = pad h = filter h / 2
* resource:
* DSP block: 
*/
__kernel
__attribute((reqd_work_group_size(BS_RELU, 1, 1)))
void relu (__global float *restrict data_input,
           __global float *restrict data_output) {
    int index       = get_global_id(0);
    
    float value     = data_input[index];
    data_output[index] = (value < 0) ? 0 : value;
}
