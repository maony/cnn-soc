#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

void gen_data(int w, int h, int k) {
    float *pw       = (float *)malloc(k*k*sizeof(float));
    float *px       = (float *)malloc(w*h*sizeof(float));
    
    int w_o = (w - k) + 1;
    int h_o = (h - k) + 1;
    float *py       = (float *)malloc(w_o * h_o * sizeof(float));
    float *pz       = (float *)malloc(w_o * h_o * sizeof(float));
    
    for(int r = 0; r < k*k; r++)
        *(pw + r) = r / k + r % k;//rand() / (float)RAND_MAX;

    for(int r = 0; r < w*h; r++)
        *(px + r) = r / w + r % w;//rand() / (float)RAND_MAX;

    for(int r = 0; r < w_o * h_o; r++)
        *(py + r) = r / w_o + r % w_o;//rand() / (float)RAND_MAX;
    
    for(int r = 0; r < w_o * h_o; r++)
        *(pz + r) = 0;

    for(int r = 0; r < h_o; r++) {
        for(int c = 0; c < w_o; c++) {
            int index = r * w_o + c; 
            for(int m = 0; m < k; m++) {
                for(int n = 0; n < k; n++) {
                    float temp;
                    temp = (*(pw + m*k + n)) * (*(px + (r+m)*w + n + c));
                    *(pz + index) += temp;
                    if( r == 0 && c == 0 && m == 0)
                        printf("r:%d, c:%d, m:%d, d:%8x, inc:%8x\n", r, c, m, *(int *)(pz+index), *(int *)&temp);
                }
                if( r == 0 && c == 0 && m == 0)
                    printf("r:%d, c:%d, m:%d, d:%8x\n", r, c, m, *(int *)(pz+index));
            }
            *(pz + index) += *(py + index);
        }
    }

    int *pbin;

    pbin = (int *)pw;
    FILE *fp = fopen("pw.txt", "w");
    for(int r = 0; r < k; r++) {
        for(int c = 0; c < k; c++) {
            fprintf(fp, "%8x\t", *(pbin+r*k+c));
        }
        fprintf(fp, "\n");
    }
    fclose(fp);

    pbin = (int *)px;
    fp = fopen("px.txt", "w");
    for(int r = 0; r < h; r++) {
        for(int c = 0; c < w; c++) {
            fprintf(fp, "%8x\t", *(pbin+r*w+c));
        }
        fprintf(fp, "\n");
    }
    fclose(fp);

    pbin = (int *)py;
    fp = fopen("py.txt", "w");
    for(int r = 0; r < h_o; r++) {
        for(int c = 0; c < w_o; c++) {
            fprintf(fp, "%8x\t", *(pbin+r*w_o+c));
        }
        fprintf(fp, "\n");
    }
    fclose(fp);

    pbin = (int *)pz;
    fp = fopen("pz.txt", "w");
    for(int r = 0; r < h_o; r++) {
        for(int c = 0; c < w_o; c++) {
            fprintf(fp, "%8x\t", *(pbin+r*w_o+c));
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
#if 1
    printf("data weight-----------\n");
    for(int r = 0; r < k; r++) {
        for(int c = 0; c < k; c++) {
            printf("%5.3f  ", *(pw + r * k + c));
        }
        printf("\n");
    }

    printf("data origin-----------\n");
    for(int r = 0; r < h; r++) {
        for(int c = 0; c < w; c++) {
            printf("%5.3f  ", *(px + r * w + c));
        }
        printf("\n");
    }

    printf("data bias-----------\n");
    for(int r = 0; r < h_o; r++) {
        for(int c = 0; c < w_o; c++) {
            printf("%5.3f  ", *(py + r * w_o + c));
        }
        printf("\n");
    }

    printf("data result-----------\n");
    for(int r = 0; r < h_o; r++) {
        for(int c = 0; c < w_o; c++) {
            printf("%5.3f  ", *(pz + r * w_o + c));
        }
        printf("\n");
    }
#endif

    free(pw);
    free(px);
    free(py);
    free(pz);

}

void parse_data(const char *file, int w, int h) {
    FILE *fp = fopen(file, "r");
    int *pdata = (int *)malloc(w*h*sizeof(int));

    for(int i = 0; i < w*h; i++)
        fscanf(fp, "%x", pdata+i);
    float *pfloat = (float *)pdata;

    printf("parse result-----------\n");
    for(int r = 0; r < h; r++) {
        for(int c = 0; c < w; c++) {
            printf("%5.3f  ", (float)*(pfloat + r * w + c));
        }
        printf("\n");
    }

    free(pdata);
    fclose(fp);
}

int main(int argc, char **argv) {
    int w, h, k;

    w = 8;
    if( argc > 1 )
        w = atoi(argv[1]);
    h = 8;
    if( argc > 2 )
        h = atoi(argv[2]);
    k = 3;
    if( argc > 3 )
        k = atoi(argv[3]);
    
    parse_data("pz.txt", (w-k)+1, (h-k)+1);
    gen_data(w, h, k);

    return 0;
}
