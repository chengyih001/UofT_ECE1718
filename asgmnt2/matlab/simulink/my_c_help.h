float convolution_layer_pre (unsigned short int filter_dim, unsigned short int i, 
                             unsigned short int j, unsigned char img[1120], float conv_w[5][49], 
                             float conv_b[5][784], float temp_weights[49], float temp_inputs[49]);

void max_pooling_layer (float sig_layer[5][784], float max_pooling[5][784], float dense_input[980]);
