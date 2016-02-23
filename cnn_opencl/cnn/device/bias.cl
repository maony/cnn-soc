/*
* bias 
* stride w = stride h = 1 
* filter w = filter h
* pad w = pad h = filter h / 2
* resource:
* DSP block: 
*/

__kernel
__attribute((reqd_work_group_size(BS_BIAS, 1, 1)))
void bias (__global float *restrict data,
           __global float *restrict bias,
           const int column) {
    int index       = get_global_id(0);
    int i;
    float value     = bias[index];
    int offset      = index * column;

    for(i = 0; i < column; i++)
        data[offset + i] = data[offset + i] + value; 
}
