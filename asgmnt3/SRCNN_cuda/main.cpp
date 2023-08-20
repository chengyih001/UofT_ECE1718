#include <iostream>
using namespace std;

#include <sstream>
#include <fstream>

typedef float d_type;

void _padding(
    d_type *in, d_type *in_padded,
    int N,
    int C, int H_in, int W_in,
    int padding
) {
    const int H_pad = H_in + padding * 2;
    const int W_pad = W_in + padding * 2;
    const int MAX_pad = N * C * H_pad * W_pad;
    
    // calloc has already set all in_pad to zero
    int n, c, h, w, i = 0;
    for (n=0; n < N; n++) {
        for (c=0; c < C; c++) {
            for (h=0; h < H_in; h++) {
                for (w=0; w < W_in; w++) {
                    // in_padded[n][c][h+padding][w+padding] = in[n][c][h][w]
                    in_padded[n*(C*H_pad*W_pad) + c*(H_pad*W_pad) + (h+padding)*(W_pad) + (w+padding)] = in[i++];
                }
            }
        }
    }
}

void CONV_forward_pass(
    d_type *in, d_type *out, d_type *weight, d_type *bias,
    int N,
    int C, int H_in, int W_in,
    int F, int H_w, int W_w,
    int H_out, int W_out,
    int stride, int padding
) {
    // pad input
    const int H_pad = H_in + padding * 2;
    const int W_pad = W_in + padding * 2;

    d_type *in_padded = (d_type *)calloc(N * C * H_pad * W_pad, sizeof(d_type));
    _padding(in, in_padded, N, C, H_in, W_in, padding);

    // conv operation
    int n, f, h, w, c, hh, ww, i=0;
    for (n=0; n < N; n++) {
        for (f=0; f < F; f++) {
            for (h=0; h < H_out; h++) {
                for (w=0; w < W_out; w++) {
                    
                    out[i] = bias[f];
                    for (c=0; c < C; c++) {
                        for (hh=0; hh < H_w; hh++) {
                            for (ww=0; ww < W_w; ww++) {
                                // out[n, f, h, w] += weight[f, c, hh, ww] * in_padded[n, c, h*stride+hh, w*stride+ww]
                                out[i] += weight[f*(C*H_w*W_w) + c*(H_w*W_w) + hh*(W_w) + ww] * in_padded[n*(C*H_pad*W_pad) + c*(H_pad*W_pad) + (h*stride+hh)*(W_pad) + (w*stride+ww)];
                            }
                        }
                    }
                    ++i;

                }
            }
        }
    }
    free(in_padded);
}

void RELU_forward_pass(
    d_type *in, d_type *out,
    int N, int C, int H, int W
) {
    // Relu::in = in;
    const int num = N*C*H*W;
    int i=0;
    
    while (i < num) {
        out[i] = in[i] > 0 ? in[i] : 0;
        ++i;
    }
}

void read_params(string file_path, d_type *param) {
    ifstream csvread;
    csvread.open(file_path, ios::in);
    if(csvread) {
        string s;
        int i=0;
        while(getline(csvread, s, '\n')) {
            param[i++] = stof(s);
        }
        csvread.close();
    }
    else{
        cerr << "Unable to read parameter!" << endl;
    exit (EXIT_FAILURE);
    }
}

void save_img(string file_path, d_type *param, const int length) {
    ofstream outfile;
	outfile.open(file_path);
    for (int i=0; i < length; i++) {
        outfile << param[i] <<endl;
    }
	outfile.close();
}

int main() {
    d_type *conv1_weight = (d_type *)calloc(64*3*9*9, sizeof(d_type));
    d_type *conv1_bias = (d_type *)calloc(64, sizeof(d_type));
    d_type *conv2_weight = (d_type *)calloc(64*32*1*1, sizeof(d_type));
    d_type *conv2_bias = (d_type *)calloc(32, sizeof(d_type));
    d_type *conv3_weight = (d_type *)calloc(32*3*5*5, sizeof(d_type));
    d_type *conv3_bias = (d_type *)calloc(3, sizeof(d_type));

    read_params("/home/chengyih001/Documents/ECE1718/A3/asgmnt3_local/SRCNN_cpp_simple/params/conv1_weight.csv", conv1_weight);
    read_params("/home/chengyih001/Documents/ECE1718/A3/asgmnt3_local/SRCNN_cpp_simple/params/conv1_bias.csv", conv1_bias);
    read_params("/home/chengyih001/Documents/ECE1718/A3/asgmnt3_local/SRCNN_cpp_simple/params/conv2_weight.csv", conv2_weight);
    read_params("/home/chengyih001/Documents/ECE1718/A3/asgmnt3_local/SRCNN_cpp_simple/params/conv2_bias.csv", conv2_bias);
    read_params("/home/chengyih001/Documents/ECE1718/A3/asgmnt3_local/SRCNN_cpp_simple/params/conv3_weight.csv", conv3_weight);
    read_params("/home/chengyih001/Documents/ECE1718/A3/asgmnt3_local/SRCNN_cpp_simple/params/conv3_bias.csv", conv3_bias);

    d_type *img = (d_type *)calloc(3*288*352, sizeof(d_type));
    d_type *res_img = (d_type *)calloc(3*288*352, sizeof(d_type));

    read_params("/home/chengyih001/Documents/ECE1718/A3/asgmnt3_local/SRCNN_cpp_simple/input_image/img0.csv", img);

    d_type *temp_buff = (d_type *)calloc(1*64*284*348, sizeof(d_type));
    d_type *temp_buff2 = (d_type *)calloc(1*32*288*352, sizeof(d_type));

    CONV_forward_pass(img, temp_buff, conv1_weight, conv1_bias,
                        1, 3, 288, 352, 64, 9, 9, 284, 348, 1, 2);
    RELU_forward_pass(temp_buff, temp_buff,
                        1, 64, 284, 348);
    
    CONV_forward_pass(temp_buff, temp_buff2, conv2_weight, conv2_bias,
                        1, 64, 284, 348, 32, 1, 1, 288, 352, 1, 2);
    RELU_forward_pass(temp_buff2, temp_buff2,
                        1, 32, 288, 352);

    CONV_forward_pass(temp_buff2, res_img, conv3_weight, conv3_bias,
                        1, 32, 288, 352, 3, 5, 5, 288, 352, 1, 2);

    save_img("/home/chengyih001/Documents/ECE1718/A3/asgmnt3_local/SRCNN_cpp_simple/output_image/img0_res.csv", res_img, 3*288*352);

    free(conv1_weight);
    free(conv1_bias);
    free(conv2_weight);
    free(conv2_bias);
    free(conv3_weight);
    free(conv3_bias);

    free(img);
    free(res_img);
    free(temp_buff);
    free(temp_buff2);
}