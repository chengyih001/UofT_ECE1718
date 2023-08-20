#include <iostream>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <fstream>

using namespace std;
const int filter_size=7;
const double eta=0.01;
const int batch_size=200;

unsigned char data_train[60000][784];
unsigned char data_test[10000][784];
unsigned char label_train[60000];
unsigned char label_test[10000];

double conv_w[5][7][7];
double conv_b[5][28][28];
double conv_layer[5][28][28];
double sig_layer[5][28][28];
char max_pooling[5][28][28];
double max_layer[5][14][14];

double dense_input[980];
double dense_w[980][120];
double dense_b[120];
double dense_sum[120];
double dense_sigmoid[120];
double dense_w2[120][10];
double dense_b2[10];
double dense_sum2[10];
double dense_softmax[10];

double dw2[120][10];
double db2[10];
double dw1[980][120];
double db1[120];

double dw_max[5][28][28];
double dw_conv[5][7][7];
double db_conv[5][28][28];


/* ************************************************************ */
#define NUM_DECIMAL_IN_BINARY 20 
int double_to_fixed_point(double x) {
    if (NUM_DECIMAL_IN_BINARY == 0)
        return (int)x;
    long temp = x * (1 << NUM_DECIMAL_IN_BINARY);
    return (int)temp;
}

double fixed_point_to_double(int x) {
    if (NUM_DECIMAL_IN_BINARY == 0)
        return (double)x;
    return (double)x / (double)(1 << NUM_DECIMAL_IN_BINARY);
}

int fixed_point_mul(int a, int b) {
    if (NUM_DECIMAL_IN_BINARY == 0)
        return a * b;
    long long temp = (long long)a * (long long)b / ((1 << NUM_DECIMAL_IN_BINARY) / 2);
    return int(temp/2 + temp%2);
}

int fixed_point_div(int a, int b) {
    if (NUM_DECIMAL_IN_BINARY == 0)
        return a / b;
    long long temp = (long long) a * (((long long)1 << NUM_DECIMAL_IN_BINARY) * 2) / b;
    return int(temp/2 + temp%2);
}

int hw_exp(int x) {
    int x_sq = fixed_point_mul(x, x);
    int ans;
    if (x < -608) ans = fixed_point_mul(0, x) + fixed_point_mul(0, x_sq) + 0;
    else if (-608 <= x && x < -576) ans = fixed_point_mul(0, x) + fixed_point_mul(0, x_sq) + 0;
    else if (-576 <= x && x < -544) ans = fixed_point_mul(0, x) + fixed_point_mul(0, x_sq) + 0;
    else if (-544 <= x && x < -512) ans = fixed_point_mul(0, x) + fixed_point_mul(0, x_sq) + 1;
    else if (-512 <= x && x < -480) ans = fixed_point_mul(0, x) + fixed_point_mul(0, x_sq) + 1;
    else if (-480 <= x && x < -448) ans = fixed_point_mul(0, x) + fixed_point_mul(0, x_sq) + 2;
    else if (-448 <= x && x < -416) ans = fixed_point_mul(1, x) + fixed_point_mul(0, x_sq) + 2;
    else if (-416 <= x && x < -384) ans = fixed_point_mul(1, x) + fixed_point_mul(0, x_sq) + 3;
    else if (-384 <= x && x < -352) ans = fixed_point_mul(1, x) + fixed_point_mul(0, x_sq) + 5;
    else if (-352 <= x && x < -320) ans = fixed_point_mul(2, x) + fixed_point_mul(0, x_sq) + 7;
    else if (-320 <= x && x < -288) ans = fixed_point_mul(3, x) + fixed_point_mul(0, x_sq) + 9;
    else if (-288 <= x && x < -256) ans = fixed_point_mul(5, x) + fixed_point_mul(0, x_sq) + 13;
    else if (-256 <= x && x < -224) ans = fixed_point_mul(7, x) + fixed_point_mul(1, x_sq) + 18;
    else if (-224 <= x && x < -192) ans = fixed_point_mul(11, x) + fixed_point_mul(1, x_sq) + 24;
    else if (-192 <= x && x < -160) ans = fixed_point_mul(15, x) + fixed_point_mul(2, x_sq) + 31;
    else if (-160 <= x && x < -128) ans = fixed_point_mul(22, x) + fixed_point_mul(3, x_sq) + 39;
    else if (-128 <= x && x < -96) ans = fixed_point_mul(31, x) + fixed_point_mul(6, x_sq) + 48;
    else if (-96 <= x && x < -64) ans = fixed_point_mul(41, x) + fixed_point_mul(9, x_sq) + 56;
    else if (-64 <= x && x < -32) ans = fixed_point_mul(53, x) + fixed_point_mul(15, x_sq) + 62;
    else if (-32 <= x && x < 0) ans = fixed_point_mul(63, x) + fixed_point_mul(25, x_sq) + 64;
    else if (0 <= x && x < 32) ans = fixed_point_mul(62, x) + fixed_point_mul(41, x_sq) + 64;
    else if (32 <= x && x < 64) ans = fixed_point_mul(35, x) + fixed_point_mul(68, x_sq) + 71;
    else if (64 <= x && x < 96) ans = fixed_point_mul(-54, x) + fixed_point_mul(112, x_sq) + 117;
    else if (96 <= x && x < 128) ans = fixed_point_mul(-274, x) + fixed_point_mul(184, x_sq) + 283;
    else if (128 <= x && x < 160) ans = fixed_point_mul(-754, x) + fixed_point_mul(303, x_sq) + 769;
    else if (160 <= x && x < 192) ans = fixed_point_mul(-1744, x) + fixed_point_mul(500, x_sq) + 2014;
    else if (192 <= x && x < 224) ans = fixed_point_mul(-3701, x) + fixed_point_mul(825, x_sq) + 4965;
    else if (224 <= x && x < 256) ans = fixed_point_mul(-7461, x) + fixed_point_mul(1360, x_sq) + 11577;
    else if (256 <= x && x < 288) ans = fixed_point_mul(-14544, x) + fixed_point_mul(2242, x_sq) + 25798;
    else if (288 <= x && x < 320) ans = fixed_point_mul(-27675, x) + fixed_point_mul(3697, x_sq) + 55447;
    else if (320 <= x && x < 352) ans = fixed_point_mul(-51723, x) + fixed_point_mul(6095, x_sq) + 115755;
    else if (352 <= x && x < 384) ans = fixed_point_mul(-95325, x) + fixed_point_mul(10049, x_sq) + 235998;
    else if (384 <= x && x < 416) ans = fixed_point_mul(-173733, x) + fixed_point_mul(16567, x_sq) + 471819;
    else if (416 <= x && x < 448) ans = fixed_point_mul(-313752, x) + fixed_point_mul(27315, x_sq) + 927945;
    else if (448 <= x && x < 480) ans = fixed_point_mul(-562323, x) + fixed_point_mul(45035, x_sq) + 1799825;
    else if (480 <= x && x < 512) ans = fixed_point_mul(-1001364, x) + fixed_point_mul(74250, x_sq) + 3449529;
    else if (512 <= x && x < 544) ans = fixed_point_mul(-1773389, x) + fixed_point_mul(122417, x_sq) + 6543405;
    else if (544 <= x && x < 576) ans = fixed_point_mul(-3125655, x) + fixed_point_mul(201832, x_sq) + 12300617;
    else if (576 <= x && x < 608) ans = fixed_point_mul(-5486099, x) + fixed_point_mul(332765, x_sq) + 22940155;
    else if (608 <= x) ans = fixed_point_mul(-9593687, x) + fixed_point_mul(548636, x_sq) + 42481615;
    return ans;
}

int hw_sigmoid(int x) {
    int x_sq = fixed_point_mul(x, x);
    int ans;
    if (-64 < x && x < 64) ans = fixed_point_mul(15, x) + 32; 
    else if (-128 <= x && x <= -64) ans = fixed_point_mul(3, x_sq) + fixed_point_mul(19, x) + 33; 
    else if (-192 <= x && x < -128) ans = fixed_point_mul(2, x_sq) + fixed_point_mul(14, x) + 28; 
    else if (-256 <= x && x < -192) ans = fixed_point_mul(1, x_sq) + fixed_point_mul(8, x) + 19; 
    else if (-320 <= x && x < -256) ans = fixed_point_mul(0, x_sq) + fixed_point_mul(4, x) + 11; 
    else if (-322 <= x && x < -320) ans = 0; 
    else if (-333 <= x && x < -322) ans = 0; 
    else if (-346 <= x && x < -333) ans = 0; 
    else if (-362 <= x && x < -346) ans = 0; 
    else if (-384 <= x && x < -362) ans = 0; 
    else if (-418 <= x && x < -384) ans = 0; 
    else if (-486 <= x && x < -418) ans = 0; 
    else if (x < -486) ans = 0; 
    else if (64 <= x && x < 128) ans = fixed_point_mul(-3, x_sq) + fixed_point_mul(19, x) + 31; 
    else if (128 <= x && x < 192) ans = fixed_point_mul(-2, x_sq) + fixed_point_mul(14, x) + 36; 
    else if (192 <= x && x < 256) ans = fixed_point_mul(-1, x_sq) + fixed_point_mul(8, x) + 45; 
    else if (256 <= x && x < 320) ans = fixed_point_mul(0, x_sq) + fixed_point_mul(4, x) + 53; 
    else if (320 <= x && x < 321) ans = 64; 
    else if (321 <= x && x < 332) ans = 64; 
    else if (332 <= x && x < 345) ans = 64; 
    else if (345 <= x && x < 361) ans = 64; 
    else if (361 <= x && x < 382) ans = 64; 
    else if (382 <= x && x < 414) ans = 64; 
    else if (414 <= x && x < 483) ans = 64; 
    else if (483 <= x) ans = 64;
    return ans;
}

/* Helper functions */
double sigmoid(double x) {
        if (x>500) x=500;
        if (x<-500) x=-500;
        return 1/(1+exp(-x));
}
double d_sigmoid(double x) {
        double sig = sigmoid(x);
        return sig*(1-sig);
}
double softmax_den(double *x, int len) {
        double val =0;
        for (int i=0; i<len; i++) {
                val += exp(x[i]);
        }
        return val;
}
int hw_softmax_den(int* x, int len) {
    int val = 0;
    for (int i = 0; i < len; i++) {
        val += hw_exp(x[i]);
    }
    return val;
}

void initialise_weights() {
        for (int i=0; i<5; i++) {
                for (int j=0; j<28; j++) {
                        for (int k=0; k<28; k++) {
                                if (j<7 && k<7) {
                                        conv_w[i][j][k] = 2*double(rand())/RAND_MAX-1;
                                }
                                conv_b[i][j][k] = 2*double(rand())/RAND_MAX-1;
                        }
                }
        }

        for (int i=0; i<980; i++) {
                for (int j=0; j<120; j++) {
                        dense_w[i][j] = 2*double(rand()) / RAND_MAX-1;
                }
        }
        for (int i=0; i<120; i++) {
                dense_b[i] = 2*double(rand()) / RAND_MAX-1;
        }

        for (int i=0; i<120; i++) {
                for (int j=0; j<10; j++) {
                        dense_w2[i][j] = 2*double(rand())/RAND_MAX-1;
                }
        }
        for (int i=0; i<10; i++) {
                dense_b2[i] = 2*double(rand())/RAND_MAX-1;
        }
}
/* ************************************************************ */

/* ************************************************************ */
/* Forward Pass */
void forward_pass(unsigned char img[][32]) {

        // Convolution Operation + Sigmoid Activation
        for (int filter_dim=0; filter_dim<5; filter_dim++) {
                for (int i=0; i<28; i++) {
                        for (int j=0; j<28; j++) {
                                max_pooling[filter_dim][i][j] = 0;

                                conv_layer[filter_dim][i][j] = 0;
                                sig_layer[filter_dim][i][j] = 0;
                                for (int k=0; k<filter_size; k++) {
                                        for (int l=0; l<filter_size; l++) {
                                                conv_layer[filter_dim][i][j] += img[i+k+1][j+l-2]*conv_w[filter_dim][k][l];
                                        }
                                }
                                sig_layer[filter_dim][i][j] = sigmoid(conv_layer[filter_dim][i][j] + conv_b[filter_dim][i][j]);
                        }
                }
        }

        // MAX Pooling (max_pooling, max_layer)
        double cur_max =0;
        int max_i=0, max_j=0;
        for (int filter_dim=0; filter_dim<5; filter_dim++) {
                for (int i=0; i<28; i+=2) {
                        for (int j=0; j<28; j+=2) {
                                max_i=i;
                                max_j=j;
                                cur_max=sig_layer[filter_dim][i][j];
                                for (int k=0; k<2; k++) {
                                        for (int l=0; l<2; l++) {
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
                                dense_input[k] = max_layer[filter_dim][i][j];
                                k++;
                        }
                }
        }

        // Dense Layer
        for (int i=0; i<120; i++) {
                dense_sum[i] = 0;
                dense_sigmoid[i] = 0;
                for (int j=0; j<980; j++) {
                        dense_sum[i] += dense_w[j][i] * dense_input[j];
                }
                dense_sum[i] += dense_b[i];
                dense_sigmoid[i] = sigmoid(dense_sum[i]);
        }

        // Dense Layer 2
        for (int i=0; i<10; i++) {
                dense_sum2[i]=0;
                for (int j=0; j<120; j++) {
                        dense_sum2[i] += dense_w2[j][i] * dense_sigmoid[j];
                }
                dense_sum2[i] += dense_b2[i];
        }

        // Softmax Output
        double den = softmax_den(dense_sum2, 10);
        for (int i=0; i<10; i++) {
                dense_softmax[i] = exp(dense_sum2[i])/den;
        }
}

void forward_pass_fixed_point(unsigned char img[][32]) {
        // local variables for fixed point values
        // inputs
        int img_fixd_point[35][32];
        int conv_w_fixd_point[5][7][7];
        int conv_b_fixd_point[5][28][28];
        int dense_w_fixd_point[980][120];
        int dense_b_fixd_point[120];
        int dense_w2_fixd_point[120][10];
        int dense_b2_fixd_point[10];
        // local
        int conv_layer_fixd_point[5][28][28];
        int sig_layer_fixd_point[5][28][28];
        int max_layer_fixd_point[5][14][14];
        int dense_sum2_fixd_point[10];
        // outputs
        int max_pooling_fixd_point[5][28][28];
        int dense_input_fixd_point[980];
        int dense_sum_fixd_point[120];
        int dense_sigmoid_fixd_point[120];
        int dense_softmax_fixd_point[10];

        // convert all inputs into fix point values
        unsigned char * uchar_ptr;
        char * char_ptr;
        double * double_ptr;
        int * int_ptr;
        uchar_ptr = &img[0][0];
        int_ptr = &img_fixd_point[0][0];
        for (int i =0; i < 35*32; i++)
            *int_ptr++ = double_to_fixed_point((double)*uchar_ptr++);
        double_ptr = &conv_w[0][0][0];
        int_ptr = &conv_w_fixd_point[0][0][0];
        for (int i =0; i < 5*7*7; i++)
            *int_ptr++ = double_to_fixed_point(*double_ptr++);
        double_ptr = &conv_b[0][0][0];
        int_ptr = &conv_b_fixd_point[0][0][0];
        for (int i =0; i < 5*28*28; i++)
            *int_ptr++ = double_to_fixed_point(*double_ptr++);
        double_ptr = &dense_w[0][0];
        int_ptr = &dense_w_fixd_point[0][0];
        for (int i =0; i < 980*120; i++)
            *int_ptr++ = double_to_fixed_point(*double_ptr++);
        double_ptr = &dense_b[0];
        int_ptr = &dense_b_fixd_point[0];
        for (int i =0; i < 120; i++)
            *int_ptr++ = double_to_fixed_point(*double_ptr++);
        double_ptr = &dense_w2[0][0];
        int_ptr = &dense_w2_fixd_point[0][0];
        for (int i =0; i < 120*10; i++)
            *int_ptr++ = double_to_fixed_point(*double_ptr++);
        double_ptr = &dense_b2[0];
        int_ptr = &dense_b2_fixd_point[0];
        for (int i =0; i < 10; i++)
            *int_ptr++ = double_to_fixed_point(*double_ptr++);

        // Convolution Operation + Sigmoid Activation
        for (int filter_dim=0; filter_dim<5; filter_dim++) {
                for (int i=0; i<28; i++) {
                        for (int j=0; j<28; j++) {
                                max_pooling_fixd_point[filter_dim][i][j] = 0;

                                conv_layer_fixd_point[filter_dim][i][j] = 0;
                                sig_layer_fixd_point[filter_dim][i][j] = 0;
                                for (int k=0; k<filter_size; k++) {
                                        for (int l=0; l<filter_size; l++) {
                                                conv_layer_fixd_point[filter_dim][i][j] += fixed_point_mul(img_fixd_point[i+k+1][j+l-2],conv_w_fixd_point[filter_dim][k][l]);
                                        }
                                }
                                sig_layer_fixd_point[filter_dim][i][j] = hw_sigmoid(conv_layer_fixd_point[filter_dim][i][j] + conv_b_fixd_point[filter_dim][i][j]);
                        }
                }
        }

        // MAX Pooling (max_pooling, max_layer)
        int cur_max =0;
        int max_i=0, max_j=0;
        for (int filter_dim=0; filter_dim<5; filter_dim++) {
                for (int i=0; i<28; i+=2) {
                        for (int j=0; j<28; j+=2) {
                                max_i=i;
                                max_j=j;
                                cur_max=sig_layer_fixd_point[filter_dim][i][j];
                                for (int k=0; k<2; k++) {
                                        for (int l=0; l<2; l++) {
                                                if (sig_layer_fixd_point[filter_dim][i+k][j+l] > cur_max) {
                                                        max_i = i+k;
                                                        max_j = j+l;
                                                        cur_max = sig_layer_fixd_point[filter_dim][max_i][max_j];
                                                }
                                        }
                                }
                                max_pooling_fixd_point[filter_dim][max_i][max_j] = double_to_fixed_point((double)1);
                                max_layer_fixd_point[filter_dim][i/2][j/2] = cur_max;
                        }
                }
        }

        int k=0;
        for (int filter_dim=0;filter_dim<5;filter_dim++) {
                for (int i=0;i<14;i++) {
                        for (int j=0;j<14;j++) {
                                dense_input_fixd_point[k] = max_layer_fixd_point[filter_dim][i][j];
                                k++;
                        }
                }
        }

        // Dense Layer
        for (int i=0; i<120; i++) {
                dense_sum_fixd_point[i] = 0;
                dense_sigmoid_fixd_point[i] = 0;
                for (int j=0; j<980; j++) {
                        dense_sum_fixd_point[i] += fixed_point_mul(dense_w_fixd_point[j][i] , dense_input_fixd_point[j]);
                }
                dense_sum_fixd_point[i] += dense_b_fixd_point[i];
                dense_sigmoid_fixd_point[i] = hw_sigmoid(dense_sum_fixd_point[i]);
        }

        // Dense Layer 2
        for (int i=0; i<10; i++) {
                dense_sum2_fixd_point[i]=0;
                for (int j=0; j<120; j++) {
                        dense_sum2_fixd_point[i] += fixed_point_mul(dense_w2_fixd_point[j][i] , dense_sigmoid_fixd_point[j]);
                }
                dense_sum2_fixd_point[i] += dense_b2_fixd_point[i];
        }

        // Softmax Output
        int den = hw_softmax_den(dense_sum2_fixd_point, 10);
        for (int i=0; i<10; i++) {
                dense_softmax_fixd_point[i] = fixed_point_div(hw_exp(dense_sum2_fixd_point[i]),den);
        }

        // convert all outputs into double values
        double_ptr = &dense_softmax[0];
        int_ptr = &dense_softmax_fixd_point[0];
        for (int i =0; i < 10; i++) {
            *double_ptr++ = fixed_point_to_double(*int_ptr++);
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
void backward_pass(double *y_hat, int *y, unsigned char img[][32]) {
        double delta4[10];
        for (int i=0; i<10; i++) {
                delta4[i] = y_hat[i] - y[i]; // Derivative of Softmax + Cross entropy
                db2[i] = delta4[i]; // Bias Changes
        }

        // Calculate Weight Changes for Dense Layer 2
        for (int i=0; i<120; i++) {
                for (int j=0; j<10; j++) {
                        dw2[i][j] = dense_sigmoid[i]*delta4[j];
                }
        }

        // Delta 3
        double delta3[120];
        for (int i=0; i<120; i++) {
                delta3[i] = 0;
                for (int j=0; j<10; j++) {
                        delta3[i] += dense_w2[i][j]*delta4[j];
                }
                delta3[i] *= d_sigmoid(dense_sum[i]);
        }
        for (int i=0; i<120; i++) db1[i] = delta3[i]; // Bias Weight change

        // Calculate Weight Changes for Dense Layer 1
        for (int i=0; i<980; i++) {
                for (int j=0; j<120; j++) {
                        dw1[i][j] = dense_input[i]*delta3[j];
                }
        }

        // Delta2
        double delta2[980];
        for (int i=0; i<980; i++) {
                delta2[i] = 0;
                for (int j=0; j<120; j++) {
                        delta2[i] += dense_w[i][j]*delta3[j];
                }
                delta2[i] *= d_sigmoid(dense_input[i]);
        }

        // Calc back-propagated max layer dw_max
        int k=0;
        for (int filter_dim=0; filter_dim<5; filter_dim++) {
                for (int i=0; i<28; i+= 2) {
                        for (int j=0; j<28; j+= 2) {
                                for (int l=0; l<2; l++) {
                                        for (int m=0; m<2; m++) {
                                                if (max_pooling[filter_dim][i+l][j+m] == 1) dw_max[filter_dim][i][j] = delta2[k];
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
                for (int i=0; i<5; i++) {
                        for (int j=0; j<5; j++) {
                                dw_conv[filter_dim][i][j] = 0;
                        }
                }
        }

        // Calculate Weight Changes for Conv Layer
        for (int filter_dim=0; filter_dim<5; filter_dim++) {
                for (int i=0; i<26; i++) {
                        for (int j=0; j<26; j++) {
                                double cur_val = dw_max[filter_dim][i][j];
                                for (int k=0; k<5; k++) {
                                        for (int l=0; l<5; l++) {
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
        csvread.open("/cad2/ece1718s/mnist_train.csv", ios::in);
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
        csvread.open("/cad2/ece1718s/mnist_test.csv", ios::in);
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

void give_img(unsigned char* vec , unsigned char img[][32]) {
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
        double max_val = dense_softmax[0];
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
                        unsigned char img[35][32];
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

        cout << "Start Testing." << endl;
        for (int i=0; i<val_len; i++) {
                unsigned char img[35][32];
                give_img(data_test[i], img);
                forward_pass_fixed_point(img);
                int pre = give_prediction();
                confusion_mat[label_test[i]][pre]++;
                if (pre == label_test[i]) cor++;
        }
        float accu = double(cor)/val_len;
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
