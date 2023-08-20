#include <iostream>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <fstream>
#include <immintrin.h>

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
float conv_layer[5][28][28];
float sig_layer[5][28][28];
char max_pooling[5][28][28];
float max_layer[5][14][14];

float dense_input[980];
float dense_w[980][120];
float dense_b[120];
float dense_sum[120];
float dense_sigmoid[120];
float dense_w2[120][10];
float dense_b2[10];
float dense_sum2[10];
float dense_softmax[10];

float dw2[120][10];
float db2[10];
float dw1[980][120];
float db1[120];

float dw_max[5][28][28];
float dw_conv[5][7][7];
float db_conv[5][28][28];

inline void mul_arrays_AVX2(float A[], float B[], float C[], int numElements){
        __m256 a_local, b_local, c_local;
        int i;
        // do 8 elements at a time as long as it is possible
        for (i = numElements; i >= 8; i-=8) {
                a_local = _mm256_loadu_ps(A); // load 8 elements (256 bits) from A
                b_local = _mm256_loadu_ps(B); // load 8 elements (256 bits) from B
                c_local = _mm256_mul_ps(a_local, b_local); // calculate 8 sums at once
                _mm256_storeu_ps(C, c_local); // store all 8 results
                A = A + 8; // advance A pointer by 8 elements (32 bytes, 256 bits)
                B = B + 8; // advance B pointer by 8 elements (32 bytes, 256 bits)
                C = C + 8; // advance C pointer by 8 elements (32 bytes, 256 bits)
        } 
        // finish the last elements by padding
        float temp_A[8], *temp_A_ptr = temp_A;
        float temp_B[8];
        int num_left = i;
        for (; i > 0; i--) {
                temp_A[i-1] = A[i-1];
                temp_B[i-1] = B[i-1];
        }
        a_local = _mm256_loadu_ps(temp_A);
        b_local = _mm256_loadu_ps(temp_B);
        c_local = _mm256_mul_ps(a_local, b_local);
        _mm256_storeu_ps(temp_A, c_local);
        for (; num_left>0; num_left--)
                *C++ = *temp_A_ptr++;
}

inline void sub_arrays_AVX2(float A[], float B[], float C[], int numElements){
        __m256 a_local, b_local, c_local;
        int i;
        // do 8 elements at a time as long as it is possible
        for (i = numElements; i >= 8; i-=8) {
                a_local = _mm256_loadu_ps(A); // load 8 elements (256 bits) from A
                b_local = _mm256_loadu_ps(B); // load 8 elements (256 bits) from B
                c_local = _mm256_sub_ps(a_local, b_local); // calculate 8 sums at once
                _mm256_storeu_ps(C, c_local); // store all 8 results
                A = A + 8; // advance A pointer by 8 elements (32 bytes, 256 bits)
                B = B + 8; // advance B pointer by 8 elements (32 bytes, 256 bits)
                C = C + 8; // advance C pointer by 8 elements (32 bytes, 256 bits)
        } 
        // finish the last elements by padding
        float temp_A[8], *temp_A_ptr = temp_A;
        float temp_B[8];
        int num_left = i;
        for (; i > 0; i--) {
                temp_A[i-1] = A[i-1];
                temp_B[i-1] = B[i-1];
        }
        a_local = _mm256_loadu_ps(temp_A);
        b_local = _mm256_loadu_ps(temp_B);
        c_local = _mm256_sub_ps(a_local, b_local);
        _mm256_storeu_ps(temp_A, c_local);
        for (; num_left>0; num_left--)
                *C++ = *temp_A_ptr++;
}

inline void add_arrays_AVX2(float A[], float B[], float C[], int numElements){
        __m256 a_local, b_local, c_local;
        int i;
        // do 8 elements at a time as long as it is possible
        for (i = numElements; i >= 8; i-=8) {
                a_local = _mm256_loadu_ps(A); // load 8 elements (256 bits) from A
                b_local = _mm256_loadu_ps(B); // load 8 elements (256 bits) from B
                c_local = _mm256_add_ps(a_local, b_local); // calculate 8 sums at once
                _mm256_storeu_ps(C, c_local); // store all 8 results
                A = A + 8; // advance A pointer by 8 elements (32 bytes, 256 bits)
                B = B + 8; // advance B pointer by 8 elements (32 bytes, 256 bits)
                C = C + 8; // advance C pointer by 8 elements (32 bytes, 256 bits)
        } 
        // finish the last elements by padding
        float temp_A[8], *temp_A_ptr = temp_A;
        float temp_B[8];
        int num_left = i;
        for (; i > 0; i--) {
                temp_A[i-1] = A[i-1];
                temp_B[i-1] = B[i-1];
        }
        a_local = _mm256_loadu_ps(temp_A);
        b_local = _mm256_loadu_ps(temp_B);
        c_local = _mm256_add_ps(a_local, b_local);
        _mm256_storeu_ps(temp_A, c_local);
        for (; num_left>0; num_left--)
                *C++ = *temp_A_ptr++;
}

inline void transpose8_ps(__m256 *row0, __m256 *row1, __m256 *row2, __m256 *row3, __m256 *row4, __m256 *row5, __m256 *row6, __m256 *row7) {
    __m256 __t0, __t1, __t2, __t3, __t4, __t5, __t6, __t7;
    __m256 __tt0, __tt1, __tt2, __tt3, __tt4, __tt5, __tt6, __tt7;
    __t0 = _mm256_unpacklo_ps(*row0, *row1);
    __t1 = _mm256_unpackhi_ps(*row0, *row1);
    __t2 = _mm256_unpacklo_ps(*row2, *row3);
    __t3 = _mm256_unpackhi_ps(*row2, *row3);
    __t4 = _mm256_unpacklo_ps(*row4, *row5);
    __t5 = _mm256_unpackhi_ps(*row4, *row5);
    __t6 = _mm256_unpacklo_ps(*row6, *row7);
    __t7 = _mm256_unpackhi_ps(*row6, *row7);
    __tt0 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(1,0,1,0));
    __tt1 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(3,2,3,2));
    __tt2 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(1,0,1,0));
    __tt3 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(3,2,3,2));
    __tt4 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(1,0,1,0));
    __tt5 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(3,2,3,2));
    __tt6 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(1,0,1,0));
    __tt7 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(3,2,3,2));
    *row0 = _mm256_permute2f128_ps(__tt0, __tt4, 0x20);
    *row1 = _mm256_permute2f128_ps(__tt1, __tt5, 0x20);
    *row2 = _mm256_permute2f128_ps(__tt2, __tt6, 0x20);
    *row3 = _mm256_permute2f128_ps(__tt3, __tt7, 0x20);
    *row4 = _mm256_permute2f128_ps(__tt0, __tt4, 0x31);
    *row5 = _mm256_permute2f128_ps(__tt1, __tt5, 0x31);
    *row6 = _mm256_permute2f128_ps(__tt2, __tt6, 0x31);
    *row7 = _mm256_permute2f128_ps(__tt3, __tt7, 0x31);
}

inline void mul_arrays_trans(float A[], float B[], float C[], int height, int width){
    // A is the 2D array that need to be transposed
    // B is the 1D array that to be multiplied
    int i, j;
    float * A_ptr = A, * C_ptr = C;
    float * B_ptr = B;
    __m256 row0, row1, row2, row3, row4, row5, row6, row7, b_local;
    for (i = width; i >= 8; i-=8) {
        for (j = height; j >= 8; j-=8) {
            b_local = _mm256_loadu_ps(B_ptr);
            row0 = _mm256_loadu_ps(A_ptr + 0*width);
            row1 = _mm256_loadu_ps(A_ptr + 1*width);
            row2 = _mm256_loadu_ps(A_ptr + 2*width);
            row3 = _mm256_loadu_ps(A_ptr + 3*width);
            row4 = _mm256_loadu_ps(A_ptr + 4*width);
            row5 = _mm256_loadu_ps(A_ptr + 5*width);
            row6 = _mm256_loadu_ps(A_ptr + 6*width);
            row7 = _mm256_loadu_ps(A_ptr + 7*width);
            transpose8_ps(&row0, &row1, &row2, &row3, &row4, &row5, &row6, &row7);
            row0 = _mm256_mul_ps(row0, b_local);
            row1 = _mm256_mul_ps(row1, b_local);
            row2 = _mm256_mul_ps(row2, b_local);
            row3 = _mm256_mul_ps(row3, b_local);
            row4 = _mm256_mul_ps(row4, b_local);
            row5 = _mm256_mul_ps(row5, b_local);
            row6 = _mm256_mul_ps(row6, b_local);
            row7 = _mm256_mul_ps(row7, b_local);
            transpose8_ps(&row0, &row1, &row2, &row3, &row4, &row5, &row6, &row7);
            _mm256_storeu_ps(C_ptr + 0*width, row0);
            _mm256_storeu_ps(C_ptr + 1*width, row1);
            _mm256_storeu_ps(C_ptr + 2*width, row2);
            _mm256_storeu_ps(C_ptr + 3*width, row3);
            _mm256_storeu_ps(C_ptr + 4*width, row4);
            _mm256_storeu_ps(C_ptr + 5*width, row5);
            _mm256_storeu_ps(C_ptr + 6*width, row6);
            _mm256_storeu_ps(C_ptr + 7*width, row7);
            if (j - 8 >= 0) {
                A_ptr = A_ptr + 8 * width;
                B_ptr = B_ptr + 8;
                C_ptr = C_ptr + 8 * width;
            }
        }
        B_ptr = B;
        A_ptr = A + (width - i + 8);
        C_ptr = C + (width - i + 8);
    }
    // finish the last elements
    for (int end_i = 0; end_i < width; end_i++) {
        for (int end_j = height - j; end_j < height; end_j++) {
            C_ptr = C + (end_i + end_j * width);
            *C_ptr = A[end_i + end_j * width] * B[end_j];
        }
    }
    // finish the last elements
    for (int end_i = width - i; end_i < width; end_i++) {
        for (int end_j = 0; end_j < height - j; end_j++) {
            C_ptr = C + (end_i + end_j * width);
            *C_ptr = A[end_i + end_j * width] * B[end_j];
        }
    }
}

float sum_array(float A[], int numElements) {
    if (numElements < 16) {
        float sum = 0;
        for (int i=0; i<numElements; i++)
            sum += A[i];
        return sum;
    } else {
        if (numElements % 2 == 0) { // even number of elements
            float * result = (float *)malloc(sizeof(float) * numElements/2);
            add_arrays_AVX2(&A[0], &A[numElements/2], result, numElements/2);
            sum_array(result, numElements/2);
        } else { // odd number of elements
            float * result = (float *)malloc(sizeof(float) * numElements/2);
            add_arrays_AVX2(&A[0], &A[numElements/2], result, numElements/2);
            result[0] += A[numElements-1];
            sum_array(result, numElements/2);
        }
    }
}

/* ************************************************************ */
/* Helper functions */
float sigmoid(float x) {
        if (x>500) x=500;
        if (x<-500) x=-500;
        return 1/(1+exp(-x));
}
float d_sigmoid(float x) {
        float sig = sigmoid(x);
        return sig*(1-sig);
}
// a SIMD version of d_sigmoid, not yet tested
inline void array_d_sigmoid(float array_in[], float array_out[], int numElements) {
    float * temp = (float *)malloc(sizeof(float) * numElements);
    float * sig = (float *)malloc(sizeof(float) * numElements);
    for (int i=0; i<numElements; i++) {
        sig[i] = sigmoid(array_in[i]);
    }
    std::fill_n(temp, numElements, 1);
    sub_arrays_AVX2(&temp[0], &sig[0], array_out, numElements);
    mul_arrays_AVX2(&sig[0], array_out, array_out, numElements);
}
float softmax_den(float *x, int len) {
        float val =0;
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
                                        conv_w[i][j][k] = 2*float(rand())/RAND_MAX-1;
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
        float cur_max =0;
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
        float den = softmax_den(dense_sum2, 10);
        for (int i=0; i<10; i++) {
                dense_softmax[i] = exp(dense_sum2[i])/den;
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
void backward_pass(float *y_hat, int *y, unsigned char img[][32]) {
        float delta4[10];
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
        float delta3[120];
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
        float delta2[980];
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
                                float cur_val = dw_max[filter_dim][i][j];
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
                forward_pass(img);
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
