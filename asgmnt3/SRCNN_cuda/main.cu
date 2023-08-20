#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sstream>
#include <fstream>
using namespace std;

typedef float d_type;

void _padding(
    d_type *in, d_type *in_padded,
    int N,
    int C, int H_in, int W_in,
    int padding
) {
    const int H_pad = H_in + padding * 2;
    const int W_pad = W_in + padding * 2;
    // const int MAX_pad = N * C * H_pad * W_pad;
    
    // calloc has already set all in_padded to zero
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

__global__ void convolution_kernel(float *input, float *kernel, float *output,
                                    int N, int C, int H_in, int W_in,
                                    int F, int H_w, int W_w, int H_out, int W_out,
                                    int stride, int H_pad, int W_pad)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N*F*H_out*W_out) {
        int w = idx % W_out;
        int h = (idx / W_out) % H_out;
        int f = (idx / (H_out*W_out)) % F;
        int n = idx / (F*H_out*W_out);

        float sum = 0.0;
        for (int c = 0; c < C; c++) {
            for (int hh = 0; hh < H_w; hh++) {
                for (int ww = 0; ww < W_w; ww++) {
                    int h_in = h * stride + hh - H_pad;
                    int w_in = w * stride + ww - W_pad;
                    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                        int input_idx = n*C*H_in*W_in + c*H_in*W_in + h_in*W_in + w_in;
                        int kernel_idx = f*C*H_w*W_w + c*H_w*W_w + hh*W_w + ww;
                        sum += input[input_idx] * kernel[kernel_idx];
                    }
                }
            }
        }
        int output_idx = n*F*H_out*W_out + f*H_out*W_out + h*W_out + w;
        output[output_idx] = sum;
    }
}

void conv(float *input, float *kernel, float *output,
            int N, int C, int H_in, int W_in,
            int F, int H_w, int W_w, int H_out, int W_out,
            int stride, int H_pad, int W_pad)
{
    int num_blocks = (N*F*H_out*W_out + 255) / 256;
    int num_threads = 256;
    convolution_kernel<<<num_blocks, num_threads>>>(input, kernel, output, 
                                                    N, C, H_in, W_in, 
                                                    F, H_w, W_w, H_out, W_out, 
                                                    stride, H_pad, W_pad);
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



    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        printf("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return 1;
    }



    // CONV_forward_pass(img, temp_buff, conv1_weight, conv1_bias,
    //                     1, 3, 288, 352, 64, 9, 9, 284, 348, 1, 2);

    // pad input
    const int H_pad = 288 + 2 * 2;
    const int W_pad = 352 + 2 * 2;
    d_type *in_padded = (d_type *)calloc(1 * 3 * H_pad * W_pad, sizeof(d_type));
    _padding(img, in_padded, 1, 3, 288, 352, 2);

    int inputWidth = W_pad;
    int inputHeight = H_pad;
    // int inputWidth = 352;
    // int inputHeight = 288;
    int inputDepth = 3;
    int kernelWidth = 9;
    int kernelHeight = 9;
    int kernelDepth = 3;
    int numFilters = 64;

    // allocate memory on device for input, output, and kernel data
    float *d_input, *d_output, *d_kernel;
    cudaMalloc((void **)&d_input, inputWidth * inputHeight * inputDepth * sizeof(float));
    cudaMalloc((void **)&d_output, 1*64*284*348 * sizeof(float));
    cudaMalloc((void **)&d_kernel, kernelWidth * kernelHeight * kernelDepth * numFilters * sizeof(float));

    // copy input and kernel data from host to device
    cudaMemcpy(d_input, in_padded, inputWidth * inputHeight * inputDepth * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, conv1_weight, kernelWidth * kernelHeight * kernelDepth * numFilters * sizeof(float), cudaMemcpyHostToDevice);

    // dim3 block(16, 16, 1);
    // dim3 grid((outputWidth + block.x - 1) / block.x, (outputHeight + block.y - 1) / block.y, numFilters);
    // convolution3D<<<grid, block>>>(d_input, d_kernel, d_output, inputDepth, inputHeight, inputWidth, kernelDepth, kernelHeight, kernelWidth, numFilters, outputHeight, outputWidth);

    conv(d_input, d_kernel, d_output,
                        1, 3, inputHeight, inputWidth, 64, 9, 9, 284, 348, 1, 2, 2);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        printf("convolution3D launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        printf("cudaDeviceSynchronize returned error code %d after launching convolution3D!\n", cudaStatus);
        return 1;
    }

    // copy output data from device to host
    cudaMemcpy(temp_buff, d_output, 1*64*284*348 * sizeof(float), cudaMemcpyDeviceToHost);

    // free memory on device
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);

    int n, f, h, w;
    int i=0;
    for (n=0; n < 1; n++) {
        for (f=0; f < 64; f++) {
            for (h=0; h < 284; h++) {
                for (w=0; w < 348; w++) {
                    temp_buff[i++] += conv1_bias[f];
                }
            }
        }
    }

    RELU_forward_pass(temp_buff, temp_buff,
                        1, 64, 284, 348);



    // CONV_forward_pass(temp_buff, temp_buff2, conv2_weight, conv2_bias,
    //                     1, 64, 284, 348, 32, 1, 1, 288, 352, 1, 2);

    // pad input
    const int H_pad_2 = 284 + 2 * 2;
    const int W_pad_2 = 348 + 2 * 2;
    d_type *in_padded_2 = (d_type *)calloc(1 * 64 * H_pad_2 * W_pad_2, sizeof(d_type));
    _padding(temp_buff, in_padded_2, 1, 64, 284, 348, 2);

    inputWidth = W_pad_2;
    inputHeight = H_pad_2;
    // inputWidth = 348;
    // inputHeight = 284;
    inputDepth = 64;
    kernelWidth = 1;
    kernelHeight = 1;
    kernelDepth = 64;
    numFilters = 32;

    // allocate memory on device for input, output, and kernel data
    cudaMalloc((void **)&d_input, inputWidth * inputHeight * inputDepth * sizeof(float));
    cudaMalloc((void **)&d_output, 1*32*288*352 * sizeof(float));
    cudaMalloc((void **)&d_kernel, kernelWidth * kernelHeight * kernelDepth * numFilters * sizeof(float));

    // copy input and kernel data from host to device
    cudaMemcpy(d_input, in_padded_2, inputWidth * inputHeight * inputDepth * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, conv2_weight, kernelWidth * kernelHeight * kernelDepth * numFilters * sizeof(float), cudaMemcpyHostToDevice);

    // dim3 grid2((outputWidth + block.x - 1) / block.x, (outputHeight + block.y - 1) / block.y, numFilters);
    // convolution3D<<<grid2, block>>>(d_input, d_kernel, d_output, inputDepth, inputHeight, inputWidth, kernelDepth, kernelHeight, kernelWidth, numFilters, outputHeight, outputWidth);

    conv(d_input, d_kernel, d_output,
                        1, 64, H_pad_2, inputWidth, 32, 1, 1, 288, 352, 1, 2, 2);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        printf("convolution3D launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        printf("cudaDeviceSynchronize returned error code %d after launching convolution3D!\n", cudaStatus);
        return 1;
    }

    // copy output data from device to host
    cudaMemcpy(temp_buff2, d_output, 1*32*288*352 * sizeof(float), cudaMemcpyDeviceToHost);

    // free memory on device
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);

    i=0;
    for (n=0; n < 1; n++) {
        for (f=0; f < 32; f++) {
            for (h=0; h < 288; h++) {
                for (w=0; w < 352; w++) {
                    temp_buff2[i++] += conv2_bias[f];
                }
            }
        }
    }

    RELU_forward_pass(temp_buff2, temp_buff2,
                        1, 32, 288, 352);



    // CONV_forward_pass(temp_buff2, res_img, conv3_weight, conv3_bias,
    //                     1, 32, 288, 352, 3, 5, 5, 288, 352, 1, 2);

    // pad input
    const int H_pad_3 = 288 + 2 * 2;
    const int W_pad_3 = 352 + 2 * 2;
    d_type *in_padded_3 = (d_type *)calloc(1 * 32 * H_pad_3 * W_pad_3, sizeof(d_type));
    _padding(temp_buff2, in_padded_3, 1, 32, 288, 352, 2);

    inputWidth = W_pad_3;
    inputHeight = H_pad_3;
    // inputWidth = 352;
    // inputHeight = 288;
    inputDepth = 32;
    kernelWidth = 5;
    kernelHeight = 5;
    kernelDepth = 32;
    numFilters = 3;

    // allocate memory on device for input, output, and kernel data
    cudaMalloc((void **)&d_input, inputWidth * inputHeight * inputDepth * sizeof(float));
    cudaMalloc((void **)&d_output, 1*3*288*352 * sizeof(float));
    cudaMalloc((void **)&d_kernel, kernelWidth * kernelHeight * kernelDepth * numFilters * sizeof(float));

    // copy input and kernel data from host to device
    cudaMemcpy(d_input, in_padded_3, inputWidth * inputHeight * inputDepth * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, conv3_weight, kernelWidth * kernelHeight * kernelDepth * numFilters * sizeof(float), cudaMemcpyHostToDevice);

    // dim3 grid3((outputWidth + block.x - 1) / block.x, (outputHeight + block.y - 1) / block.y, numFilters);
    // convolution3D<<<grid3, block>>>(d_input, d_kernel, d_output, inputDepth, inputHeight, inputWidth, kernelDepth, kernelHeight, kernelWidth, numFilters, outputHeight, outputWidth);

    conv(d_input, d_kernel, d_output,
                        1, 32, inputHeight, inputWidth, 3, 5, 5, 288, 352, 1, 2, 2);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        printf("convolution3D launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        printf("cudaDeviceSynchronize returned error code %d after launching convolution3D!\n", cudaStatus);
        return 1;
    }
    
    // copy output data from device to host
    cudaMemcpy(res_img, d_output, 1*3*288*352 * sizeof(float), cudaMemcpyDeviceToHost);

    // free memory on device
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);

    i=0;
    for (n=0; n < 1; n++) {
        for (f=0; f < 3; f++) {
            for (h=0; h < 288; h++) {
                for (w=0; w < 352; w++) {
                    res_img[i++] += conv3_bias[f];
                }
            }
        }
    }


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