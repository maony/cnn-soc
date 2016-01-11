/*
    use channel to copmute convolution




*/
#pragma OPENCL EXTENSION cl_altera_channels:enable

typedef float16 T;
#define DEPTH   1

channel T DATA_IN_00           __attribute__((depth(DEPTH)));
channel T DATA_OUT             __attribute__((depth(DEPTH)));

kernel __attribute__((task))
void conv_read_16x16(global T * restrict data_input, uint channel_num) {
    global T *ptr_input;
    T data;

#define MEM_WRITE(i, j)                                 \
    do {                                                \
        data = *ptr_input++;                            \
        write_channel_altera(DATA_IN_##i##j, data);     \
    }while(0)
    
    unsigned char iter = 0;

    for(unsigned char j = 0; j < 14; j++) {
        ptr_input = data_input + j;
        iter = 0;
        for(uint i = 0; i < channel_num * 3; i++) {
            MEM_WRITE(0, 0);
            iter += 1;
            if(iter == 3) {
                ptr_input += 13;
                iter = 0;
            }
        }
    }
#undef MEM_WRITE
}

kernel __attribute__((task))
void conv_compute_16x16(global float * restrict filter, uint channel_num, uint offset_filter) {
    T       data;
    T       row = (T)0.0;
    
    float   data_filter[3];
    global float   *ptr_filter = filter + offset_filter;
    
    for(unsigned char i = 0; i < 14; i++) {
        ptr_filter = filter + offset_filter;
        row = (T)(0);
        for(unsigned char j = 0; j < channel_num * 3; j++) {
            data = read_channel_altera(DATA_IN_00);
            data_filter[0] = *ptr_filter++;
            data_filter[1] = *ptr_filter++;
            data_filter[2] = *ptr_filter++;
#define ROW_COMPUTE(i)      \
            do {                                    \
                row.s0 += data.s0 * data_filter[i];    \
                row.s1 += data.s1 * data_filter[i];    \
                row.s2 += data.s2 * data_filter[i];    \
                row.s3 += data.s3 * data_filter[i];    \
                row.s4 += data.s4 * data_filter[i];    \
                row.s5 += data.s5 * data_filter[i];    \
                row.s6 += data.s6 * data_filter[i];    \
                row.s7 += data.s7 * data_filter[i];    \
                row.s8 += data.s8 * data_filter[i];    \
                row.s9 += data.s9 * data_filter[i];    \
                row.sa += data.sa * data_filter[i];    \
                row.sb += data.sb * data_filter[i];    \
                row.sc += data.sc * data_filter[i];    \
                row.sd += data.sd * data_filter[i];    \
            } while(0)
            ROW_COMPUTE(0);
            data = data.s123456789abcdef0;
            ROW_COMPUTE(1);
            data = data.s123456789abcdef0;
            ROW_COMPUTE(2);
#undef ROW_COMPUTE
        }
        write_channel_altera(DATA_OUT, row);
    }
}

kernel __attribute__((task))
void conv_write_16x16(global T * restrict data_output, uint offset) {
    T data;
    global T *ptr_out = data_output + offset;

    for(unsigned char i = 0; i < 14; i++) {
        data        = read_channel_altera(DATA_OUT);
        ptr_out[i]  = data;
    }
}

