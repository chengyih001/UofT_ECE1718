#include <iostream>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <fstream>
#include "vadd.h"

using namespace std;
const int filter_size=7;
const float eta=0.01;
const int batch_size=200;
unsigned char data_train[60000][784];
unsigned char data_test[10000][784];
unsigned char label_train[60000];
unsigned char label_test[10000];
float conv_w[5][7][7];
float conv_b[5][28][28];
float conv_layer_tb[5][28][28];
float sig_layer_tb[5][28][28];
char max_pooling_tb[5][28][28];
float max_layer_tb[5][14][14];
float dense_input_tb[980];
float dense_w[980][120];
float dense_b[120];
float dense_sum_tb[120];
float dense_sigmoid_tb[120];
float dense_w2[120][10];
float dense_b2[10];
float dense_sum_tb2[10];
float dense_softmax[10];
float dw2[120][10];
float db2[10];
float dw1[980][120];
float db1[120];
float dw_max[5][28][28];
float dw_conv[5][7][7];
float db_conv[5][28][28];
/* ************************************************************ */
/* Helper functions */
//float sigmoid(float x) {
//        if (x>500) x=500;
//        if (x<-500) x=-500;
//        return 1/(1+exp(-x));
//}
float d_sigmoid(float x) {
        float sig = sigmoid(x);
        return sig*(1-sig);
}
//float softmax_den(float *x, int len) {
//        float val =0;
//        for (int i=0; i<len; i++) {
//                val += exp(x[i]);
//        }
//        return val;
//}
void initialise_weights() {
        for (int i=0; i<5; i++) {
                for (int j=0; j<28; j++) {
                        for (int k=0; k<28; k++) {
                                if (j<7 && k<7) {
                                        conv_w[i][j][k] = 
2*float(rand())/RAND_MAX-1;
                                }
                                conv_b[i][j][k] = 2*float(rand())/RAND_MAX-1;
                        }
                }
        }
        for (int i=0; i<980; i++) {
                for (int j=0; j<120; j++) {
                        dense_w[i][j] = 2*float(rand()) / RAND_MAX-1;
                }
        }
        for (int i=0; i<120; i++) {
                dense_b[i] = 2*float(rand()) / RAND_MAX-1;
        }
        for (int i=0; i<120; i++) {
                for (int j=0; j<10; j++) {
                        dense_w2[i][j] = 2*float(rand())/RAND_MAX-1;
                }
        }
        for (int i=0; i<10; i++) {
                dense_b2[i] = 2*float(rand())/RAND_MAX-1;
        }
}
/* ************************************************************ */
/* ************************************************************ */
/* Forward Pass */
void forward_pass(float img[][32]) {
        // Convolution Operation + Sigmoid Activation
        for (int filter_dim=0; filter_dim<5; filter_dim++) {
                for (int i=0; i<28; i++) {
                        for (int j=0; j<28; j++) {
                                max_pooling_tb[filter_dim][i][j] = 0;
                                conv_layer_tb[filter_dim][i][j] = 0;
                                sig_layer_tb[filter_dim][i][j] = 0;
                                for (int k=0; k<filter_size; k++) {
                                        for (int l=0; l<filter_size; l++) {
                                                conv_layer_tb[filter_dim][i][j] += img[i+k+1][j+l-2]*conv_w[filter_dim][k][l];
                                        }
                                }
                                sig_layer_tb[filter_dim][i][j] = sigmoid(conv_layer_tb[filter_dim][i][j] + conv_b[filter_dim][i][j]);
                        }
                }
        }
        // MAX Pooling (max_pooling_tb, max_layer_tb)
        float cur_max =0;
        int max_i=0, max_j=0;
        for (int filter_dim=0; filter_dim<5; filter_dim++) {
                for (int i=0; i<28; i+=2) {
                        for (int j=0; j<28; j+=2) {
                                max_i=i;
                                max_j=j;
                                cur_max=sig_layer_tb[filter_dim][i][j];
                                for (int k=0; k<2; k++) {
                                        for (int l=0; l<2; l++) {
                                                if (sig_layer_tb[filter_dim][i+k][j+l]> cur_max) {
                                                        max_i = i+k;
                                                        max_j = j+l;
                                                        cur_max = sig_layer_tb[filter_dim][max_i][max_j];
                                                }
                                        }
                                }
                                max_pooling_tb[filter_dim][max_i][max_j] = 1;
                                max_layer_tb[filter_dim][i/2][j/2] = cur_max;
                        }
                }
        }
        int k=0;
        for (int filter_dim=0;filter_dim<5;filter_dim++) {
                for (int i=0;i<14;i++) {
                        for (int j=0;j<14;j++) {
                                dense_input_tb[k] = max_layer_tb[filter_dim][i][j];
                                k++;
                        }
                }
        }
        // Dense Layer
        for (int i=0; i<120; i++) {
                dense_sum_tb[i] = 0;
                dense_sigmoid_tb[i] = 0;
                for (int j=0; j<980; j++) {
                        dense_sum_tb[i] += dense_w[j][i] * dense_input_tb[j];
                }
                dense_sum_tb[i] += dense_b[i];
                dense_sigmoid_tb[i] = sigmoid(dense_sum_tb[i]);
        }
        // Dense Layer 2
        for (int i=0; i<10; i++) {
                dense_sum_tb2[i]=0;
                for (int j=0; j<120; j++) {
                        dense_sum_tb2[i] += dense_w2[j][i] * dense_sigmoid_tb[j];
                }
                dense_sum_tb2[i] += dense_b2[i];
        }
        // Softmax Output
        float den = softmax_den(dense_sum_tb2, 10);
        for (int i=0; i<10; i++) {
                dense_softmax[i] = exp(dense_sum_tb2[i])/den;
        }
}
void update_weights() {
        for (int i=0; i<120; i++) {
                dense_b[i] -= eta*db1[i];
                for (int j=0; j<10; j++) {
                        dense_b2[j] -= eta*db2[j];
                        dense_w2[i][j] -= eta*dw2[i][j];
                }
                for (int k=0; k<980; k++) {
                        dense_w[k][i] -= eta*dw1[k][i];
                }
        }
        for (int i=0; i<5; i++) {
                for (int k=0; k<7; k++) {
                        for (int j=0; j<7; j++) {
                                conv_w[i][k][j] -= eta*dw_conv[i][k][j];
                        }
                }
                for (int l=0; l<28; l++) {
                        for (int m=0; m<28; m++) {
                                conv_b[i][l][m] -= eta*db_conv[i][l][m];
                        }
                }
        }
}
/* ************************************************************ */
/* ************************************************************ */
/* Backward Pass */
void backward_pass(float *y_hat, int *y, float img[][32]) {
        float delta4[10];
        for (int i=0; i<10; i++) {
                delta4[i] = y_hat[i] - y[i]; // Derivative of Softmax + Cross entropy
                db2[i] = delta4[i]; // Bias Changes
        }
        // Calculate Weight Changes for Dense Layer 2
        for (int i=0; i<120; i++) {
                for (int j=0; j<10; j++) {
                        dw2[i][j] = dense_sigmoid_tb[i]*delta4[j];
                }
        }
        // Delta 3
        float delta3[120];
        for (int i=0; i<120; i++) {
                delta3[i] = 0;
                for (int j=0; j<10; j++) {
                        delta3[i] += dense_w2[i][j]*delta4[j];
                }
                delta3[i] *= d_sigmoid(dense_sum_tb[i]);
        }
        for (int i=0; i<120; i++) db1[i] = delta3[i]; // Bias Weight change
        // Calculate Weight Changes for Dense Layer 1
        for (int i=0; i<980; i++) {
                for (int j=0; j<120; j++) {
                        dw1[i][j] = dense_input_tb[i]*delta3[j];
                }
        }
        // Delta2
        float delta2[980];
        for (int i=0; i<980; i++) {
                delta2[i] = 0;
                for (int j=0; j<120; j++) {
                        delta2[i] += dense_w[i][j]*delta3[j];
                }
                delta2[i] *= d_sigmoid(dense_input_tb[i]);
        }
        // Calc back-propagated max layer dw_max
        int k=0;
        for (int filter_dim=0; filter_dim<5; filter_dim++) {
                for (int i=0; i<28; i+= 2) {
                        for (int j=0; j<28; j+= 2) {
                                for (int l=0; l<2; l++) {
                                        for (int m=0; m<2; m++) {
                                                if (max_pooling_tb[filter_dim][i+l][j+m] == 1) dw_max[filter_dim][i][j] = delta2[k];
                                                else dw_max[filter_dim][i][j] = 0;
                                        }
                                }
                                k++;
                        }
                }
        }
        // Calc Conv Bias Changes
        for (int filter_dim=0; filter_dim<5; filter_dim++) {
                for (int i=0; i<28; i++) {
                        for (int j=0; j<28; j++) {
                                db_conv[filter_dim][i][j] = dw_max[filter_dim][i][j];
                        }
                }
        }
        // Set Conv Layer Weight changes to 0
        for (int filter_dim=0; filter_dim<5; filter_dim++) {
                for (int i=0; i<7; i++) {
                        for (int j=0; j<7; j++) {
                                dw_conv[filter_dim][i][j] = 0;
                        }
                }
        }
        // Calculate Weight Changes for Conv Layer
        for (int filter_dim=0; filter_dim<5; filter_dim++) {
                for (int i=0; i<28; i++) {
                        for (int j=0; j<28; j++) {
                                float cur_val = dw_max[filter_dim][i][j];
                                for (int k=0; k<7; k++) {
                                        for (int l=0; l<7; l++) {
                                                dw_conv[filter_dim][k][l] += img[i+k+1][j+l-2] * cur_val;
                                        }
                                }
                        }
                }
        }
}
/* ************************************************************ */
void read_train_data() {
        ifstream csvread;
        csvread.open("data/mnist_train.csv", ios::in);
        if(csvread) {
                string s;
                int data_pt = 0;
                while(getline(csvread, s)) {
                        stringstream ss(s);
                        int pxl = 0;
                        while( ss.good() ) {
                                string substr;
                                getline(ss, substr,',');
                                if (pxl == 0) {
                                        label_train[data_pt] = stoi(substr);
                                } else {
                                        data_train[data_pt][pxl-1] = stoi(substr);
                                }
                                pxl++;
                        }
                        data_pt++;
                }
                csvread.close();
        }
        else{
                cerr << "Unable to read train data!" << endl;
        exit (EXIT_FAILURE);
        }
}
void read_test_data() {
        ifstream csvread;
        csvread.open("data/mnist_test.csv", ios::in);
        if(csvread) {
                string s;
                int data_pt = 0;
                while(getline(csvread, s)) {
                        stringstream ss(s);
                        int pxl = 0;
                        while( ss.good() ) {
                                string substr;
                                getline(ss, substr,',');
                                if (pxl == 0) {
                                        label_test[data_pt] = stoi(substr);
                                } else {
                                        data_test[data_pt][pxl-1] = stoi(substr);
                                }
                                pxl++;
                        }
                        data_pt++;
                }
                csvread.close();
        }
        else{
                cerr << "Unable to read test data!" << endl;
        exit (EXIT_FAILURE);
        }
}
void give_img(unsigned char* vec , float img[][32]) {
        int k=0;
        for (int i=0; i<35; i++) {
                for (int j=0; j<32; j++) {
                        if (i<5 || j<2 || i>32 || j>29) {
                                img[i][j] = 0;
                        } else {
                                img[i][j] = vec[k++];
                        }
                }
        }
}
void give_y(int y, int *vector_y) {
        for (int i=0; i<10; i++) vector_y[i] =0;
        vector_y[y]=1;
}
int give_prediction() {
        float max_val = dense_softmax[0];
        int max_pos = 0;
        for (int i=1; i<10; i++) {
                if (dense_softmax[i] > max_val) {
                        max_val = dense_softmax[i];
                        max_pos = i;
                }
        }
        return max_pos;
}
int main() {
        read_test_data();
        read_train_data();
        initialise_weights();
        int epoch = 500;
        int num = 0;
        cout << "Start Training." << endl;
        for (int i=0; i<epoch; i++) {
                cout << "Epoch " << i << " done." << endl;
                for (int j=0; j<batch_size; j++) {
                        num = rand()%60000;
                        float img[35][32];
                        int vector_y[10];
                        give_y(label_train[num], vector_y);
                        give_img(data_train[num], img);
                        forward_pass(img);
                        backward_pass(dense_softmax, vector_y, img);
                        update_weights();
                }
        }
        int val_len = 600;
        int cor=0;
        int confusion_mat[10][10];
        for (int i=0; i<10; i++){
                for (int j=0; j<10; j++) confusion_mat[i][j] = 0;
        }

        float krnl_conv_w[5*7*7];
        float krnl_conv_b[5*28*28];
        float krnl_dense1_w[980*120];
        float krnl_dense1_b[120];
        float krnl_dense2_w[120*10];
        float krnl_dense2_b[10];


        for (int i=0; i < 5; i++) {
            for (int j=0; j < 7; j++) {
                for (int k=0; k < 7; k++) {
                    krnl_conv_w[i*7*7 + j*7 + k] = conv_w[i][j][k];
                }
            }
        }

        for (int i=0; i < 5; i++) {
            for (int j=0; j < 28; j++) {
                for (int k=0; k < 28; k++) {
                    krnl_conv_b[i*28*28 + j*28 + k] = conv_b[i][j][k];
                }
            }
        }

        for (int i=0; i < 980; i++) {
            for (int j=0; j < 120; j++) {
                krnl_dense1_w[i*120+j] = dense_w[i][j];
            }
        }

        for (int i=0; i < 120; i++) {
            for (int j=0; j < 10; j++) {
                krnl_dense2_w[i*10+j] = dense_w2[i][j];
            }
        }

        for (int i=0; i < 120; i++) {
            krnl_dense1_b[i] = dense_b[i];
        }

        for (int i=0; i < 10; i++) {
            krnl_dense2_b[i] = dense_b2[i];
        }

        cout << "Start Testing." << endl;
        for (int i=0; i<val_len; i++) {

        		float tmp[35*32];
				float img[35][32];
				give_img(data_test[i], img);
				for (int j=0; j < 35; j++) {
					for (int k=0; k < 32; k++) {
						tmp[j*32+k] = img[j][k];
					}
				}

                vadd(tmp, dense_softmax, krnl_conv_w, krnl_conv_b, krnl_dense1_w, krnl_dense1_b, krnl_dense2_w, krnl_dense2_b);

                int pre = give_prediction();
                confusion_mat[label_test[i]][pre]++;
                if (pre == label_test[i]) cor++;
        }
        float accu = float(cor)/val_len;
        cout << "Accuracy: " << accu << endl;
        cout << "   0 1 2 3 4 5 6 7 8 9" << endl;
        for (int i=0; i<10; i++){
                cout << i << ": ";
                for (int j=0; j<10; j++) {
                        cout << confusion_mat[i][j] << " ";
                }
                cout << endl;
        }
        return 0;
}
