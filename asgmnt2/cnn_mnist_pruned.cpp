#include <iostream>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <fstream>

#pragma warning(disable : 4996)
FILE* weights_file;
int image_num;

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
double dense_w[980][10];
double dense_b[10];
double dense_sum[10];
double dense_softmax[10];

double dw1[980][10];
double db1[980];

double dw_max[5][28][28];
double dw_conv[5][7][7];
double db_conv[5][28][28];

/* ************************************************************ */
// set to -1 and double for original behavior
#define NUM_DECIMAL_IN_BINARY 8
#define CONV_DATA_TYPE int

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
                for (int j=0; j<10; j++) {
                        dense_w[i][j] = 2*double(rand()) / RAND_MAX-1;
                }
        }
        for (int i=0; i<10; i++) {
                dense_b[i] = 2*double(rand()) / RAND_MAX-1;
        }
}
/* ************************************************************ */

/* ************************************************************ */
double convolution(double *input, double *weight, double bias, int len) {
        if (NUM_DECIMAL_IN_BINARY == -1) { // if not using fixed point
                CONV_DATA_TYPE sum = (CONV_DATA_TYPE)bias;
                for (int i=0; i<len; i++)
                        sum += (CONV_DATA_TYPE)input[i] * (CONV_DATA_TYPE)weight[i];
                return (double)sum;
        } else { // if using fix point
                CONV_DATA_TYPE sum = double_to_fixed_point(bias);
                for (int i=0; i<len; i++)
                        sum += fixed_point_mul(double_to_fixed_point(input[i]), double_to_fixed_point(weight[i]));
                return fixed_point_to_double(sum);
        }
}

double convolution_trans(double *input, double *weight, double bias, int input_col_index, int input_width, int len) {
        if (NUM_DECIMAL_IN_BINARY == -1) { // if not using fixed point
                CONV_DATA_TYPE sum = (CONV_DATA_TYPE)bias;
                for (int i=0; i<len; i++)
                        sum += (CONV_DATA_TYPE)input[i * input_width + input_col_index] * (CONV_DATA_TYPE)weight[i];
                return (double)sum;
        } else { // if using fix point
                CONV_DATA_TYPE sum = double_to_fixed_point(bias);
                for (int i=0; i<len; i++)
                        sum += fixed_point_mul(double_to_fixed_point(input[i * input_width + input_col_index]), double_to_fixed_point(weight[i]));
                return fixed_point_to_double(sum);
        }
}

double convolution_scaler(double *input, double scaler, double bias, int len) {
        if (NUM_DECIMAL_IN_BINARY == -1) { // if not using fixed point
                CONV_DATA_TYPE sum = (CONV_DATA_TYPE)bias;
                for (int i=0; i<len; i++)
                        sum += (CONV_DATA_TYPE)input[i] * (CONV_DATA_TYPE)scaler;
                return (double)sum;
        } else { // if using fix point
                CONV_DATA_TYPE sum = double_to_fixed_point(bias);
                for (int i=0; i<len; i++)
                        sum += fixed_point_mul(double_to_fixed_point(input[i]), double_to_fixed_point(scaler));
                return fixed_point_to_double(sum);
        }
}

/* Forward Pass */
void forward_pass(unsigned char img[][32]) {

        // Convolution Operation + Sigmoid Activation
        for (int filter_dim=0; filter_dim<5; filter_dim++) {
                for (int i=0; i<28; i++) {
                        for (int j=0; j<28; j++) {
                                max_pooling[filter_dim][i][j] = 0;
                                conv_layer[filter_dim][i][j] = 0;
                                sig_layer[filter_dim][i][j] = 0;

                                double temp_wights[49];
                                double temp_inputs[49];
                                for (int k=0; k<filter_size; k++) {
                                        for (int l=0; l<filter_size; l++) {
                                                // conv_layer[filter_dim][i][j] += img[i+k+1][j+l-2]*conv_w[filter_dim][k][l];
                                                temp_wights[k*7 + l] = img[i+k+1][j+l-2];
                                                temp_inputs[k*7 + l] = conv_w[filter_dim][k][l];
                                        }
                                }
                                conv_layer[filter_dim][i][j] = convolution(temp_inputs, temp_wights, conv_b[filter_dim][i][j], 49);
                                sig_layer[filter_dim][i][j] = sigmoid(conv_layer[filter_dim][i][j]);
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
        for (int i=0; i<10; i++) {
                // dense_sum[i] = 0;
                // for (int j=0; j<980; j++) {
                //         dense_sum[i] += dense_w[j][i] * dense_input[j];
                // }
                // dense_sum[i] += dense_b[i];
                dense_sum[i] = convolution_trans(&dense_w[0][0], dense_input, dense_b[i], i, 10, 980);
        }

        // Softmax Output
        double den = softmax_den(dense_sum, 10);
        for (int i=0; i<10; i++) {
                dense_softmax[i] = exp(dense_sum[i])/den;
        }
}

void update_weights() {
        for (int i=0; i<10; i++) {
                dense_b[i] -= eta*db1[i];
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
                db1[i] = delta4[i]; // Bias Changes
        }

        // Calculate Weight Changes for Dense Layer 2
        for (int i=0; i<980; i++) {
                for (int j=0; j<10; j++) {
                        dw1[i][j] = dense_input[i]*delta4[j];
                }
        }

        // Delta 3
        double delta3[980];
        for (int i=0; i<980; i++) {
                // delta3[i] = 0;
                // for (int j=0; j<10; j++) {
                //         delta3[i] += dense_w[i][j]*delta4[j];
                // }
                delta3[i] = convolution(delta4, dense_w[i], 0, 10);
                delta3[i] *= d_sigmoid(dense_input[i]);
        }

        // Calc back-propagated max layer dw_max
        int k=0;
        for (int filter_dim=0; filter_dim<5; filter_dim++) {
                for (int i=0; i<28; i+= 2) {
                        for (int j=0; j<28; j+= 2) {
                                for (int l=0; l<2; l++) {
                                        for (int m=0; m<2; m++) {
                                                if (max_pooling[filter_dim][i+l][j+m] == 1) dw_max[filter_dim][i][j] = delta3[k];
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
                                double cur_val = dw_max[filter_dim][i][j];
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
        csvread.open("C:/Course Material/UofT/ECE1718/assignment2/mnist_train.csv", ios::in);
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
        csvread.open("C:/Course Material/UofT/ECE1718/assignment2/mnist_test.csv", ios::in);
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
        double* double_ptr;
        unsigned char* uchar_ptr;
        char* char_ptr;
        weights_file = fopen("weights.txt", "w");

        image_num = -1;
        
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
        
        double_ptr = &conv_w[0][0][0];
        for (int i=0; i<5*7*7; i++)
        fprintf(weights_file, "conv_w = %lf\n", *double_ptr++);
        double_ptr = &conv_b[0][0][0];
        for (int i = 0; i < 5 * 28 * 28; i++)
        fprintf(weights_file, "conv_b = %lf\n", *double_ptr++);
        double_ptr = &dense_w[0][0];
        for (int i = 0; i < 980 * 10; i++)
        fprintf(weights_file, "dense_w = %lf\n", *double_ptr++);
        double_ptr = &dense_b[0];
        for (int i = 0; i < 10; i++)
        fprintf(weights_file, "dense_b = %lf\n", *double_ptr++);
        
        cout << "Start Testing." << endl;
        for (int i=0; i<val_len; i++) {
                unsigned char img[35][32];
                give_img(data_test[i], img);
                forward_pass(img);
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
