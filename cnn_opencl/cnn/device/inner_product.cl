/*
* inner product 
* stride w = stride h = 1 
* filter w = filter h
* pad w = pad h = filter h / 2
* resource:
* DSP block: 
*/
__kernel
__attribute((reqd_work_group_size(BS_INNER_PRODUCT, 1, 1)))
void inner_product (__global float *restrict data_input,
                    __global float *restrict weight,
                    __global float *restrict bias,
                    __global float *restrict data_output,
                    const int column) {
    int index       = get_global_id(0);
    
    __global float *ptr_weight = weight + index * column;
    float value = 0;
    int i;
    
    for(i = 0; i < column; i++)
        value += data_input[i] * ptr_weight[i];

    data_output[index] = value + bias[index];
}
