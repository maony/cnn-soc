/*
    use channel to copmute convolution




*/
#pragma OPENCL EXTENSION cl_altera_channels:enable

typedef float8 T;
#define DEPTH   1

// Channel declarations
channel T DATA_IN_00           __attribute__((depth(DEPTH)));
channel T DATA_IN_01           __attribute__((depth(DEPTH)));
channel T DATA_IN_02           __attribute__((depth(DEPTH)));
channel T DATA_IN_03           __attribute__((depth(DEPTH)));
channel float DATA_OUT         __attribute__((depth(DEPTH + 1)));

#define MEM_WRITE(i, j)                                 \
    do {                                                \
        data = *ptr_frame++;                            \
        write_channel_altera(DATA_IN_##i##j, data);     \
    }while(0)

kernel __attribute__((task))
void conv_read_8x8(global T * restrict data_input, uint channel_num) {
    global T *ptr_channel, *ptr_frame;
    T data;

    for(uint i = 0; i < channel_num; i++) {
        ptr_channel = data_input + i * 8;
        for(uint j = 0; j < 3; j++) {
            ptr_frame = ptr_channel + j*2;
            MEM_WRITE(0, 0);
            MEM_WRITE(0, 1);
            MEM_WRITE(0, 2);
            MEM_WRITE(0, 3);
        }
    }
}

kernel __attribute__((task))
void conv_compute_8x8(global float * restrict filter, uint channel_num) {
    T       data;//, temp;
    float   row[6*6] = {0.0f};
    float   data_filter[9];
    global float   *ptr_filter = filter;

    for(uint i = 0; i < channel_num; i++) {
        for(uint j = 0; j < 9; j++)
            data_filter[j]  = *ptr_filter++;
        //for(uint j = 0; j < 8; j++)
        //    data[j] = read_channel_altera(DATA_IN_00);
        
        for(uint i = 0; i < 6; i += 2) {
            //row[0]         += data_filter[0] * data.s0;
            //row[1]         += data_filter[0] * data.s1;
            //row[2]         += data_filter[0] * data.s2;
            //row[4]         += data_filter[0] * data.s3;
            //row[5]         += data_filter[0] * data.s4;
            //row[6]         += data_filter[0] * data.s5;
#define ROW_COMPUTE(i, j)       \
            do {                \
                row[i+0]    += data_filter[j] * data.s0;    \
                row[i+1]    += data_filter[j] * data.s1;    \
                row[i+2]    += data_filter[j] * data.s2;    \
                row[i+3]    += data_filter[j] * data.s3;    \
                row[i+4]    += data_filter[j] * data.s4;    \
                row[i+5]    += data_filter[j] * data.s5;    \
            } while(0)
            data            = read_channel_altera(DATA_IN_00);
            mem_fence(CLK_CHANNEL_MEM_FENCE);
            ROW_COMPUTE(6*i+0, 0);
            data            = data.s12345670;
            ROW_COMPUTE(6*i+0, 1);
            data            = data.s12345670;
            ROW_COMPUTE(6*i+0, 2);
           
            data            = read_channel_altera(DATA_IN_01);
            mem_fence(CLK_CHANNEL_MEM_FENCE);
            ROW_COMPUTE(6*i+0, 3);
            ROW_COMPUTE(6*i+6, 0);
            data            = data.s12345670;
            ROW_COMPUTE(6*i+0, 4);
            ROW_COMPUTE(6*i+6, 1);
            data            = data.s12345670;
            ROW_COMPUTE(6*i+0, 5);
            ROW_COMPUTE(6*i+6, 2);
           
            data            = read_channel_altera(DATA_IN_02);
            mem_fence(CLK_CHANNEL_MEM_FENCE);
            ROW_COMPUTE(6*i+0, 6);
            ROW_COMPUTE(6*i+6, 3);
            data            = data.s12345670;
            ROW_COMPUTE(6*i+0, 7);
            ROW_COMPUTE(6*i+6, 4);
            data            = data.s12345670;
            ROW_COMPUTE(6*i+0, 8);
            ROW_COMPUTE(6*i+6, 5);
            
            data            = read_channel_altera(DATA_IN_03);
            mem_fence(CLK_CHANNEL_MEM_FENCE);
            ROW_COMPUTE(6*i+6, 6);
            data            = data.s12345670;
            ROW_COMPUTE(6*i+6, 7);
            data            = data.s12345670;
            ROW_COMPUTE(6*i+6, 8);
#undef ROW_COMPUTE
            //data[0]         = read_channel_altera(DATA_IN_00);
            //mem_fence(CLK_CHANNEL_MEM_FENCE);
            //row[i]         += data_filter[0] * data[0];
            //data[0]         = data[0].s12345670;
            //row[i]         += data_filter[1] * data[0];
            //data[0]         = data[0].s12345670;
            //row[i]         += data_filter[2] * data[0];
        
            //data[0]         = read_channel_altera(DATA_IN_01);
            //mem_fence(CLK_CHANNEL_MEM_FENCE);
            //row[i]         += data_filter[3] * data[0];
            //row[i+1]       += data_filter[0] * data[0];
            //data[0]         = data[0].s12345670;
            //row[i]         += data_filter[4] * data[0];
            //row[i+1]       += data_filter[1] * data[0];
            //data[0]         = data[0].s12345670;
            //row[i]         += data_filter[5] * data[0];
            //row[i+1]       += data_filter[2] * data[0];
           
            //data[0]         = read_channel_altera(DATA_IN_02);
            //mem_fence(CLK_CHANNEL_MEM_FENCE);
            //row[i]         += data_filter[6] * data[0];
            //row[i+1]       += data_filter[3] * data[0];
            //data[0]         = data[0].s12345670;
            //row[i]         += data_filter[7] * data[0];
            //row[i+1]       += data_filter[4] * data[0];
            //data[0]         = data[0].s12345670;
            //row[i]         += data_filter[8] * data[0];
            //row[i+1]       += data_filter[5] * data[0];

            //data[0]         = read_channel_altera(DATA_IN_03);
            //mem_fence(CLK_CHANNEL_MEM_FENCE);
            //row[i+1]       += data_filter[6] * data[0];
            //data[0]         = data[0].s12345670;
            //row[i+1]       += data_filter[7] * data[0];
            //data[0]         = data[0].s12345670;
            //row[i+1]       += data_filter[8] * data[0];
            //temp            = data[i];
            //row[i]         += data_filter[0] * temp;
            //temp            = temp.s12345670;
            //row[i]         += data_filter[1] * temp;
            //temp            = temp.s12345670;
            //row[i]         += data_filter[2] * temp;

            //temp            = data[i+1];
            //row[i]         += data_filter[0] * temp;
            //temp            = temp.s12345670;
            //row[i]         += data_filter[1] * temp;
            //temp            = temp.s12345670;
            //row[i]         += data_filter[2] * temp;

            //temp            = data[i+2];
            //row[i]         += data_filter[0] * temp;
            //temp            = temp.s12345670;
            //row[i]         += data_filter[1] * temp;
            //temp            = temp.s12345670;
            //row[i]         += data_filter[2] * temp;
        }

        // row[0]         += data_filter[0] * data[0];
        // data[0]         = data[0].s12345670;
        // row[0]         += data_filter[1] * data[0];
        // data[0]         = data[0].s12345670;
        // row[0]         += data_filter[2] * data[0];

        // row[0]         += data_filter[3] * data[1];
        // row[1]         += data_filter[0] * data[1];
        // data[1]         = data[1].s12345670;
        // row[0]         += data_filter[4] * data[1];
        // row[1]         += data_filter[1] * data[1];
        // data[1]         = data[1].s12345670;
        // row[0]         += data_filter[5] * data[1];
        // row[1]         += data_filter[2] * data[1];

        // row[0]         += data_filter[6] * data[2];
        // row[1]         += data_filter[3] * data[2];
        // row[2]         += data_filter[0] * data[2];
        // data[2]         = data[2].s12345670;
        // row[0]         += data_filter[7] * data[2];
        // row[1]         += data_filter[4] * data[2];
        // row[2]         += data_filter[1] * data[2];
        // data[2]         = data[2].s12345670;
        // row[0]         += data_filter[8] * data[2];
        // row[1]         += data_filter[5] * data[2];
        // row[2]         += data_filter[2] * data[2];
        // 
        // row[1]         += data_filter[6] * data[3];
        // row[2]         += data_filter[3] * data[3];
        // row[3]         += data_filter[0] * data[3];
        // data[3]         = data[3].s12345670;
        // row[1]         += data_filter[7] * data[3];
        // row[2]         += data_filter[4] * data[3];
        // row[3]         += data_filter[1] * data[3];
        // data[3]         = data[3].s12345670;
        // row[1]         += data_filter[8] * data[3];
        // row[2]         += data_filter[5] * data[3];
        // row[3]         += data_filter[2] * data[3];

        // row[2]         += data_filter[6] * data[4];
        // row[3]         += data_filter[3] * data[4];
        // row[4]         += data_filter[0] * data[4];
        // data[4]         = data[4].s12345670;
        // row[2]         += data_filter[7] * data[4];
        // row[3]         += data_filter[4] * data[4];
        // row[4]         += data_filter[1] * data[4];
        // data[4]         = data[4].s12345670;
        // row[2]         += data_filter[8] * data[4];
        // row[3]         += data_filter[5] * data[4];
        // row[4]         += data_filter[2] * data[4];

        // row[3]         += data_filter[6] * data[5];
        // row[4]         += data_filter[3] * data[5];
        // row[5]         += data_filter[0] * data[5];
        // data[5]         = data[5].s12345670;
        // row[3]         += data_filter[7] * data[5];
        // row[4]         += data_filter[4] * data[5];
        // row[5]         += data_filter[1] * data[5];
        // data[5]         = data[5].s12345670;
        // row[3]         += data_filter[8] * data[5];
        // row[4]         += data_filter[5] * data[5];
        // row[5]         += data_filter[2] * data[5];

        // row[4]         += data_filter[6] * data[6];
        // row[5]         += data_filter[3] * data[6];
        // data[6]         = data[6].s12345670;
        // row[4]         += data_filter[7] * data[6];
        // row[5]         += data_filter[4] * data[6];
        // data[6]         = data[6].s12345670;
        // row[4]         += data_filter[8] * data[6];
        // row[5]         += data_filter[5] * data[6];

        // row[5]         += data_filter[6] * data[7];
        // data[7]         = data[7].s12345670;
        // row[5]         += data_filter[7] * data[7];
        // data[7]         = data[7].s12345670;
        // row[5]         += data_filter[8] * data[7];
    }
  
    for(uint i = 0; i < 6*6; i++) {
        write_channel_altera(DATA_OUT, row[i]); 
        //row[i] = (T)(row[i].s012345, 0.0f, 0.0f);
        //write_channel_altera(DATA_OUT, row[i]);
        //mem_fence(CLK_CHANNEL_MEM_FENCE);
    }
}

kernel __attribute__((task))
void conv_write_8x8(global T * restrict data_output) {
    float data[6];

    for(uint i = 0; i < 6; i++) {
        for(uint j = 0; j < 6; j++)
            data[j] = read_channel_altera(DATA_OUT);
        data_output[i] = (T)(data[0], data[1], data[2], data[3], data[4], data[5], 0, 0);
        //data            = read_channel_altera(DATA_OUT);
        //mem_fence(CLK_CHANNEL_MEM_FENCE);
        //data_output[i]  = data;
    }
}

