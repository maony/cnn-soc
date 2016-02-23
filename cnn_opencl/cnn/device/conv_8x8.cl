/*
    use channel to copmute convolution




*/
#pragma OPENCL EXTENSION cl_altera_channels:enable

#ifdef NONE_VEC
typedef float   T;
#define DEPTH   8
#else
typedef float8  T;
#define DEPTH   1
#endif

// Channel declarations
channel T DATA_IN_00           __attribute__((depth(DEPTH)));
channel T DATA_OUT             __attribute__((depth(DEPTH)));

kernel __attribute__((task))
void conv_read_8x8(global T * restrict data_input, uint channel_num) {
    global T *ptr_input = data_input;
    T data;

#define MEM_WRITE(i, j)                                 \
    do {                                                \
        data = *ptr_input++;                            \
        write_channel_altera(DATA_IN_##i##j, data);     \
    }while(0)

    for(uint i = 0; i < channel_num; i++) {
#ifdef NONE_VEC
        for(uint j = 0; j < 8*8; j++)
#else
        for(uint j = 0; j < 8; j++)
#endif
            MEM_WRITE(0, 0);
    }
#undef MEM_WRITE
}

kernel __attribute__((task))
void conv_compute_8x8(global float * restrict filter, uint channel_num, uint offset_filter) {
#ifdef NONE_VEC
    T       data[8*8], temp;
    T       row[6*6] = {0.0f};
#elif defined(SHIFT)
    T       data[3];
    T       row[6];
#else
    T       data[8], temp;
    T       row[6] = {0.0};
#endif
    float   data_filter[9];
    global float   *ptr_filter = filter + offset_filter;
    uint    row_index;

    for(uint i = 0; i < channel_num; i++) {
        for(uint j = 0; j < 9; j++)
            data_filter[j]  = *ptr_filter++;
#ifdef SHIFT
        for(uint j = 0; j < 8; j++) {
            data[j] = read_channel_altera(DATA_IN_00);

        }
#elif defined(NONE_VEC)
        for(uint j = 0; j < 8*8; j++) {
            data[j] = read_channel_altera(DATA_IN_00);
        }
#define ROW_COMPUTE(i, j, k)    \
        do {                    \
            row[i+0] += data_filter[3*filter_index+j] * data[k+0];  \
            row[i+1] += data_filter[3*filter_index+j] * data[k+1];  \
            row[i+2] += data_filter[3*filter_index+j] * data[k+2];  \
            row[i+3] += data_filter[3*filter_index+j] * data[k+3];  \
            row[i+4] += data_filter[3*filter_index+j] * data[k+4];  \
            row[i+5] += data_filter[3*filter_index+j] * data[k+5];  \
        } while(0)
        
        for(uint j = 0; j < 6*3; j++) {
            uint row_index      = j / 3;
            uint filter_index   = j - row_index * 3;
            
            ROW_COMPUTE(row_index*6, 0, (row_index+filter_index)*8+0);
            ROW_COMPUTE(row_index*6, 1, (row_index+filter_index)*8+1);
            ROW_COMPUTE(row_index*6, 2, (row_index+filter_index)*8+2);
        }
#undef  ROW_COMPUTE
#else
        for(uint j = 0; j < 8; j++)
            data[j] = read_channel_altera(DATA_IN_00);
#define UNROLL_ONE_ROW
#ifdef UNROLL_ONE_ROW
        for(uint j = 0; j < 6*3; j++) {
            uint row_index      = j / 3;
            uint filter_index   = j - row_index * 3;
#define ROW_COMPUTE(i)          \
        do {                    \
            row[row_index].s0 += data_filter[3*filter_index + i] * temp.s0; \
            row[row_index].s1 += data_filter[3*filter_index + i] * temp.s1; \
            row[row_index].s2 += data_filter[3*filter_index + i] * temp.s2; \
            row[row_index].s3 += data_filter[3*filter_index + i] * temp.s3; \
            row[row_index].s4 += data_filter[3*filter_index + i] * temp.s4; \
            row[row_index].s5 += data_filter[3*filter_index + i] * temp.s5; \
            temp = temp.s12345670;                                          \
        } while(0)
            temp = data[row_index + filter_index];
            ROW_COMPUTE(0);
            ROW_COMPUTE(1);
            ROW_COMPUTE(2);
#undef ROW_COMPUTE
        }
#else
#define ROW_COMPUTE_0(i, j, k)  \
        do {                    \
            row[i].s0  += data_filter[j] * data[k].s0; \
            row[i].s1  += data_filter[j] * data[k].s1; \
            row[i].s2  += data_filter[j] * data[k].s2; \
            row[i].s3  += data_filter[j] * data[k].s3; \
            row[i].s4  += data_filter[j] * data[k].s4; \
            row[i].s5  += data_filter[j] * data[k].s5; \
        } while(0)
#define ROW_COMPUTE_1(i, j, k)      \
        do {                    \
            row[i].s0  += data_filter[j] * data[k].s1; \
            row[i].s1  += data_filter[j] * data[k].s2; \
            row[i].s2  += data_filter[j] * data[k].s3; \
            row[i].s3  += data_filter[j] * data[k].s4; \
            row[i].s4  += data_filter[j] * data[k].s5; \
            row[i].s5  += data_filter[j] * data[k].s6; \
        } while(0)
#define ROW_COMPUTE_2(i, j, k)      \
        do {                    \
            row[i].s0  += data_filter[j] * data[k].s2; \
            row[i].s1  += data_filter[j] * data[k].s3; \
            row[i].s2  += data_filter[j] * data[k].s4; \
            row[i].s3  += data_filter[j] * data[k].s5; \
            row[i].s4  += data_filter[j] * data[k].s6; \
            row[i].s5  += data_filter[j] * data[k].s7; \
        } while(0)
        for(uint i = 0; i < 6; i++) {
            ROW_COMPUTE_0(i, 0, i);
            ROW_COMPUTE_1(i, 1, i);
            ROW_COMPUTE_2(i, 2, i);
            ROW_COMPUTE_0(i, 3, i+1);
            ROW_COMPUTE_1(i, 4, i+1);
            ROW_COMPUTE_2(i, 5, i+1);
            ROW_COMPUTE_0(i, 6, i+2);
            ROW_COMPUTE_1(i, 7, i+2);
            ROW_COMPUTE_2(i, 8, i+2);
        }
#undef ROW_COMPUTE_0
#undef ROW_COMPUTE_1
#undef ROW_COMPUTE_2
#endif
#endif
    }

#ifdef NONE_VEC
    for(uint i = 0; i < 6*6; i++) {
        write_channel_altera(DATA_OUT, row[i]);    
    }
#else
    for(uint i = 0; i < 6; i++) {
        row[i] = (T)(row[i].s012345, 0.0f, 0.0f);
        write_channel_altera(DATA_OUT, row[i]);
        mem_fence(CLK_CHANNEL_MEM_FENCE);
    }
#endif
}

kernel __attribute__((task))
void conv_write_8x8(global T * restrict data_output, uint offset) {
    T data;
    global T *ptr_out = data_output + offset;
#ifdef NONE_VEC
    for(uint i = 0; i < 6; i++) {
        for(uint j = 0; j < 6; j++)
            ptr_out[j] = read_channel_altera(DATA_OUT);
        ptr_out += 8;
    }
#else
    for(uint i = 0; i < 6; i++) {
        data            = read_channel_altera(DATA_OUT);
        mem_fence(CLK_CHANNEL_MEM_FENCE);
        ptr_out[i]  = data;
    }
#endif
}

