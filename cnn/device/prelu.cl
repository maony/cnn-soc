/*
* prelu 
* stride w = stride h = 1 
* filter w = filter h
* pad w = pad h = filter h / 2
* resource:
* DSP block: 
*/

__kernel
__attribute((reqd_work_group_size(BS_PRELU, 1, 1)))
void prelu (__global float *restrict data,
            __global float *restrict slope,
            const int column) {
    int index       = get_global_id(0);
    int i;
    float value     = slope[index];
    int offset      = index * column;
    float temp;

    #pragma unroll 0
    for(i = 0; i < column; i++) {
        temp = data[offset+i];    
        data[offset + i] = (temp > 0) ? temp : temp * value; 
    }
}
