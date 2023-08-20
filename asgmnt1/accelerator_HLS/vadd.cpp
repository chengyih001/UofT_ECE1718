#include "vadd.h"

d_type conv_layer[5][28][28];
d_type sig_layer[5][28][28];
char max_pooling[5][28][28];
d_type max_layer[5][14][14];

d_type dense_input[980];
d_type dense_sum[120];
d_type dense_sigmoid[120];
d_type dense_sum2[10];
const int filter_size=7;

d_type sigmoid(d_type x) {
        if (x>500) x=500;
        if (x<-500) x=-500;
        return 1/(1+exp(-x));
}

d_type softmax_den(d_type *x, int len) {

        d_type tmp0[5];
        d_type tmp1[3];
        d_type tmp2[2];
        d_type val;

        tmp0[0] = exp(x[0]) + exp(x[1]);
        tmp0[1] = exp(x[2]) + exp(x[3]);
        tmp0[2] = exp(x[4]) + exp(x[5]);
        tmp0[3] = exp(x[6]) + exp(x[7]);
        tmp0[4] = exp(x[8]) + exp(x[9]);

        tmp1[0] = tmp0[0] + tmp0[1];
        tmp1[1] = tmp0[2] + tmp0[3];
        tmp1[2] = tmp0[4];

        tmp2[0] = tmp1[0] + tmp1[1];
        tmp2[1] = tmp1[2];

        val = tmp2[0] + tmp2[1];

        return val;
}

extern "C" {

void vadd(
            d_type* in, d_type* out,
            d_type* conv_w, d_type* conv_b,
            d_type* dense1_w, d_type* dense1_b,
            d_type* dense2_w, d_type* dense2_b
        )
{
#pragma HLS INTERFACE m_axi port = in bundle = gmem
#pragma HLS INTERFACE m_axi port = out bundle = gmem

#pragma HLS INTERFACE ap_ctrl_hs port = return

    d_type input_buffer[35*32];
    d_type output_buffer[10];
//#pragma HLS array_partition variable=input_buffer complete

    for (int i=0; i < 35*32; i++) {
        input_buffer[i] = in[i];
    }

    for (int filter_dim=0; filter_dim<5; filter_dim++) {
    	for (int i=0; i < 28; i++) {
    		for (int j=0; j < 28; j++) {
    			max_pooling[filter_dim][i][j] = 0;
				conv_layer[filter_dim][i][j] = 0;
				sig_layer[filter_dim][i][j] = 0;
    		}
    	}
    }

    d_type conv_temp[49];
//#pragma HLS array_partition variable=conv_temp complete

    for (int filter_dim=0; filter_dim<5; filter_dim++) {
            for (int i=0; i<28; i++) {
                    for (int j=0; j<28; j++) {
#pragma HLS PIPELINE
                            for (int k=0; k<filter_size; k++) {
                                    for (int l=0; l<filter_size; l++) {
#pragma HLS UNROLL
//                                            conv_layer[filter_dim][i][j] += in[(i+k+1)*32 + j+l-2]*conv_w[filter_dim*7*7 + k*7 + l];
                                    		conv_temp[k*7+l] = input_buffer[(i+k+1)*32 + j+l-2]*conv_w[filter_dim*7*7 + k*7 + l];
                                    }
                            }

                            for (int m=0; m < 49; m++) {
                            	conv_layer[filter_dim][i][j] += conv_temp[m];
                            }

                    }
            }
    }

    for (int filter_dim=0; filter_dim<5; filter_dim++) {
    	for (int i=0; i < 28; i++) {
    		for (int j=0; j < 28; j++) {
#pragma HLS PIPELINE
    			d_type tmp = conv_layer[filter_dim][i][j] + conv_b[filter_dim*28*28 + i*28 + j];
    			sig_layer[filter_dim][i][j] = sigmoid(tmp);

    		}
    	}
    }

    // MAX Pooling (max_pooling, max_layer)
    d_type cur_max =0;
    int max_i=0, max_j=0;
    for (int filter_dim=0; filter_dim<5; filter_dim++) {
            for (int i=0; i<28; i+=2) {
                    for (int j=0; j<28; j+=2) {
                            max_i=i;
                            max_j=j;
                            cur_max=sig_layer[filter_dim][i][j];
                            for (int k=0; k<2; k++) {
                                    for (int l=0; l<2; l++) {
#pragma HLS PIPELINE II=4
                                            if (sig_layer[filter_dim][i+k][j+l] > cur_max) {
                                                    max_i = i+k;
                                                    max_j = j+l;
                                                    cur_max = sig_layer[filter_dim][max_i][max_j];
                                            }
                                    }
                            }
                            max_pooling[filter_dim][max_i][max_j] = 1;
                            max_layer[filter_dim][i/2][j/2] = cur_max;
                    }
            }
    }

    int k=0;
    for (int filter_dim=0;filter_dim<5;filter_dim++) {
            for (int i=0;i<14;i++) {
                    for (int j=0;j<14;j++) {
#pragma HLS PIPELINE
                            dense_input[k] = max_layer[filter_dim][i][j];
                            k++;
                    }
            }
    }

    for (int i=0; i<120; i++) {
            dense_sum[i] = 0;
            dense_sigmoid[i] = 0;
    }

    // Dense Layer
    for (int i=0; i<120; i++) {
            for (int j=0; j<980; j++) {
#pragma HLS PIPELINE II=3
                    dense_sum[i] += dense1_w[j*120 + i] * dense_input[j];
            }

    }

    for (int i=0; i < 120; i++) {
#pragma HLS PIPELINE
    	dense_sum[i] += dense1_b[i];
        dense_sigmoid[i] = sigmoid(dense_sum[i]);
    }

    // Dense Layer 2
    for (int i=0; i < 10; i++) {
    	dense_sum2[i]=0;
    }

    for (int i=0; i<10; i++) {

            for (int j=0; j<120; j++) {
#pragma HLS PIPELINE II=3
                    dense_sum2[i] += dense2_w[j*10 + i] * dense_sigmoid[j];
            }

    }

    for (int i=0; i < 10; i++) {
#pragma HLS PIPELINE
    	dense_sum2[i] += dense2_b[i];
    }

    // Softmax Output
    d_type den = softmax_den(dense_sum2, 10);
    for (int i=0; i<10; i++) {
#pragma HLS PIPELINE
            output_buffer[i] = exp(dense_sum2[i])/den;
    }

    for (int i=0; i < 10; i++) {
#pragma HLS PIPELINE
        out[i] = output_buffer[i];
    }



}

}
