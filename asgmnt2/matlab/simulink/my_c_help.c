float convolution_layer_pre (unsigned short int filter_dim, unsigned short int i, 
                             unsigned short int j, unsigned char img[1120], float conv_w[5][49], 
                             float conv_b[5][784], float temp_weights[49], float temp_inputs[49]) {
    for (int k=0; k<7; k++) {
        for (int l=0; l<7; l++) {
            temp_weights[k*7 + l] = img[(i+k+1)*32 + (j+l-2)];
            temp_inputs[k*7 + l] = conv_w[filter_dim][k*7 + l];
        }
    }

    float bias = conv_b[filter_dim][i*28 + j];
    return bias;
}

void max_pooling_layer (float sig_layer[5][784], float max_pooling[5][784], float dense_input[980]) {
    float max_layer[5][14*14];

    // max pooling
    double cur_max =0;
    int max_i=0, max_j=0;
    for (int filter_dim=0; filter_dim<5; filter_dim++) {
        for (int i=0; i<28; i+=2) {
            for (int j=0; j<28; j+=2) {
                max_i=i;
                max_j=j;
                cur_max=sig_layer[filter_dim][i*28 + j];
                for (int k=0; k<2; k++) {
                    for (int l=0; l<2; l++) {
                        if (sig_layer[filter_dim][(i+k)*28 + (j+l)] > cur_max) {
                            max_i = i+k;
                            max_j = j+l;
                            cur_max = sig_layer[filter_dim][(max_i)*28 + max_j];
                        }
                    }
                }
                max_pooling[filter_dim][max_i*28 + max_j] = 1;
                max_layer[filter_dim][(i/2)*14 + (j/2)] = cur_max;
            }
        }
    }

    int k=0;
    for (int filter_dim=0;filter_dim<5;filter_dim++) {
        for (int i=0;i<14;i++) {
            for (int j=0;j<14;j++) {
                dense_input[k] = max_layer[filter_dim][i*28 + j];
                k++;
            }
        }
    }
}
