#include "conv.h"

Conv2d::Conv2d(
) {
    const int DEFAULT_VAL = -1;
    N = DEFAULT_VAL;
    C = DEFAULT_VAL;
    H_in = DEFAULT_VAL;
    W_in = DEFAULT_VAL;
    F = DEFAULT_VAL;
    H_w = DEFAULT_VAL;
    W_w = DEFAULT_VAL;

    stride = DEFAULT_STRIDE;
    padding = DEFAULT_PAD;

    H_out = DEFAULT_VAL;
    W_out = DEFAULT_VAL;

    in_padded = nullptr;
    weight = nullptr;
    bias = nullptr;
    
    dx = nullptr;
    dw = nullptr;
    db = nullptr;
}

Conv2d::Conv2d(
    std::unordered_map<std::string, int> param_dict,
    d_type *weight,
    d_type *bias
) {
    set_params(param_dict, weight, bias);
}

Conv2d::~Conv2d(
) {
    free(in_padded);
    free(dx);
    free(dw);
    free(db);
}

/**
 * @brief Set parameters of current convolution layer from external places
 * 
 * @param param_dict parameter dictionary containing shape of x, w, b, stride, padding etc.
 * @param weight weight data to be set to current convolution layer
 * @param bias bias data to be set to current convolution layer
 */
void Conv2d::set_params(
    std::unordered_map<std::string, int> param_dict,
    d_type *weight,
    d_type *bias
) {
    const int H_pad = H_in + padding * 2;
    const int W_pad = W_in + padding * 2;

    N = param_dict["N"];
    C = param_dict["C"];
    H_in = param_dict["H_in"];
    W_in = param_dict["W_in"];
    F = param_dict["F"];
    H_w = param_dict["H_w"];
    W_w = param_dict["W_w"];

    stride = param_dict["STRIDE"];
    padding = param_dict["PADDING"];

    H_out = 1 + int((H_in + 2 * padding - H_w) / stride);
    W_out = 1 + int((W_in + 2 * padding - W_w) / stride);

    in_padded = (d_type *)calloc(N * C * H_pad * W_pad, sizeof(d_type));
    Conv2d::weight = weight;
    Conv2d::bias = bias;

    dx = (d_type *)calloc(N * C * H_in * W_in, sizeof(d_type));
    dw = (d_type *)calloc(F * C * H_w * W_w, sizeof(d_type));
    db = (d_type *)calloc(F, sizeof(d_type));

}

/**
 * @brief Perform padding of the input
 * 
 * @param[in] in input image for current convolution layer
 * @param[out] in_padded save padded input to object's memory
 */
void Conv2d::_padding(
    d_type *in
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
                    // cout << n << " " << c << " " << h << " " << w << " " << endl;
                    // in_paddedded[n][c][h+padding][w+padding] = in[n][c][h][w]
                    in_padded[n*(C*H_pad*W_pad) + c*(H_pad*W_pad) + (h+padding)*(W_pad) + (w+padding)] = in[i++];
                }
            }
        }
    }
}

/**
 * @brief Get parameters of current convolution layer from external places
 * 
 * @return std::unordered_map<std::string, int> containing shape of x, w, b, stride, padding etc.
 */
std::unordered_map<std::string, int> Conv2d::get_params(
) {
    std::unordered_map<std::string, int> param_dict;

    param_dict["N"] = N;
    param_dict["C"] = C;
    param_dict["H_in"] = H_in;
    param_dict["W_in"] =  W_in;
    param_dict["F"] = F;
    param_dict["H_w"] = H_w; 
    param_dict["W_w"] = W_w;

    param_dict["STRIDE"] = stride;
    param_dict["PADDING"] = padding;

    param_dict["H_out"] = H_out;
    param_dict["W_out"] = W_out;

    return param_dict;
}

/**
 * @brief Convolution forward_pass
 * 
 * @param in input x for convolution forward pass
 * @return d_type* result of convolution forward pass
 */
void Conv2d::forward_pass(
    d_type *in,
    d_type *out
) {
    // pad input
    const int H_pad = H_in + padding * 2;
    const int W_pad = W_in + padding * 2;
    Conv2d::_padding(in);

    // conv operation
    int n, f, h, w, c, hh, ww, i=0;
    for (n=0; n < N; n++) {
        for (f=0; f < F; f++) {
            for (h=0; h < H_out; h++) {
                for (w=0; w < W_out; w++) {
                    
                    out[i] += bias[f];
                    for (c=0; c < C; c++) {
                        for (hh=0; hh < H_w; hh++) {
                            for (ww=0; ww < W_w; ww++) {
                                // cout << i << endl;
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
}

/**
 * @brief Convolution backward_pass
 * 
 * @param dout output of previous layer's backward pass
 * @return std::unordered_map<std::string, d_type*> containing ["dx"], ["dw"], ["db"] for gradient descent
 */
std::unordered_map<std::string, d_type*> Conv2d::backward_pass(
    d_type *dout
) {
    const int H_pad = H_in + padding * 2;
    const int W_pad = W_in + padding * 2;

    d_type *dx_padded = (d_type *)calloc(N * C * H_pad * W_pad, sizeof(d_type));

    int n, f, h, w, c, hh, ww, i=0;
    for (n=0; n < N; n++) {
        for (f=0; f < F; f++) {
            for (h=0; h < H_out; h++) {
                for (w=0; w < W_out; w++) {

                    db[f] += dout[i];
                    for (c=0; c < C; c++) {
                        for (hh=0; hh < H_w; hh++) {
                            for (ww=0; ww < W_w; ww++) {
                                dw[f*(C*H_w*W_w) + c*(H_w*W_w) + hh*(W_w) + ww] += in_padded[n*(C*H_out*W_out) + c*(H_out*W_out) + (h*stride+hh)*(W_out) + (w*stride+ww)] * dout[i];
                                dx_padded[n*(C*H_out*W_out) + c*(H_out*W_out) + (h*stride+hh)*(W_out) + (w*stride+ww)] += weight[f*(C*H_w*W_w) + c*(H_w*W_w) + hh*(W_w) + ww] * dout[i];
                            }
                        }
                    }
                    ++i;

                }
            }
        }
    }

    i=0;
    for (n=0; n < N; n++) {
        for (c=0; c < C; C++) {
            for (h=0; h < H_in; h++) {
                for (w=0; w < W_in; w++) {
                    dx[i++] = dx_padded[n*(C*H_pad*W_pad) + c*(H_pad*W_pad) + (h+padding)*W_pad + (w+padding)];
                }
            }
        }
    }

    std::unordered_map<std::string, d_type*> res_dict;
    res_dict["dx"] = dx;
    res_dict["dw"] = dw;
    res_dict["db"] = db;

    free(dx_padded);

    return res_dict;
}
