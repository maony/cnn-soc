/*
* sigmoid 
* stride w = stride h = 1 
* filter w = filter h
* pad w = pad h = filter h / 2
* resource:
* DSP block: 
*/

__kernel
__attribute((reqd_work_group_size(BS_SIGMOID, 1, 1)))
void sigmoid (__global float * data) {
    int index       = get_global_id(0);
    float value     = data[index];
    
    value = 1.0 / (1.0 + exp(-value));
    data[index] = value;
}
