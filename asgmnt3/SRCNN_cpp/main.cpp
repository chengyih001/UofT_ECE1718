#include <iostream>
using namespace std;

#include "SRCNN.h"
#include <string>
#include <sstream>
#include <fstream>

typedef float d_type;

void read_params(string file_path, d_type* param) {
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

int main() {
    d_type *conv1_weight = (d_type *)calloc(64*3*9*9, sizeof(d_type));
    d_type *conv1_bias = (d_type *)calloc(64, sizeof(d_type));
    d_type *conv2_weight = (d_type *)calloc(64*32*1*1, sizeof(d_type));
    d_type *conv2_bias = (d_type *)calloc(32, sizeof(d_type));
    d_type *conv3_weight = (d_type *)calloc(32*3*5*5, sizeof(d_type));
    d_type *conv3_bias = (d_type *)calloc(3, sizeof(d_type));

    read_params("/Users/Joey/Google_Drive/Year_1_2/ECE1718/A3/SRCNN_cpp/params/conv1_weight.csv", conv1_weight);
    read_params("/Users/Joey/Google_Drive/Year_1_2/ECE1718/A3/SRCNN_cpp/params/conv1_bias.csv", conv1_bias);
    read_params("/Users/Joey/Google_Drive/Year_1_2/ECE1718/A3/SRCNN_cpp/params/conv2_weight.csv", conv2_weight);
    read_params("/Users/Joey/Google_Drive/Year_1_2/ECE1718/A3/SRCNN_cpp/params/conv2_bias.csv", conv2_bias);
    read_params("/Users/Joey/Google_Drive/Year_1_2/ECE1718/A3/SRCNN_cpp/params/conv3_weight.csv", conv3_weight);
    read_params("/Users/Joey/Google_Drive/Year_1_2/ECE1718/A3/SRCNN_cpp/params/conv3_bias.csv", conv3_bias);

    d_type *img = (d_type *)calloc(3*288*352, sizeof(d_type));
    read_params("/Users/Joey/Google_Drive/Year_1_2/ECE1718/A3/SRCNN_cpp/input_image/img0.csv", img);

    SRCNN model;
    model.set_params(
        conv1_weight, conv1_bias,
        conv2_weight, conv2_bias,
        conv3_weight, conv3_bias
    );

    d_type *res_img = (d_type *)calloc(3*288*352, sizeof(d_type));

    model.forward_pass(img, res_img);
    // cout << (res_img[0]) << endl;

    // free(img);
    free(conv1_weight);
    free(conv1_bias);
    free(conv2_weight);
    free(conv2_bias);
    free(conv3_weight);
    free(conv3_bias);
    // free(res_img);
}