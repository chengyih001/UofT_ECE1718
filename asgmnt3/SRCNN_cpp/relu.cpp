#include "relu.h"

Relu::Relu(
) {
    const int DEFAULT_VAL = -1;
    N = DEFAULT_VAL;
    C = DEFAULT_VAL;
    H = DEFAULT_VAL;
    W = DEFAULT_VAL;

    dx = nullptr;
}

Relu::Relu(
    std::unordered_map<std::string, int> param_dict
) {
    set_params(param_dict);
}

Relu::~Relu(
) {
    free(dx);
}

/**
 * @brief Set parameters of current Relu layer from external places
 * 
 * @param param_dict parameter dictionary containing shape of x
 */
void Relu::set_params(
    std::unordered_map<std::string, int> param_dict
) {
    N = param_dict["N"];
    C = param_dict["C"];
    H = param_dict["H"];
    W = param_dict["W"];

    dx = (d_type *)calloc(N * C * H * W, sizeof(d_type));
}

/**
 * @brief Get parameters of current Relu layer from external places
 * 
 * @return std::unordered_map<std::string, int> containing parameters of current Relu layer
 */
std::unordered_map<std::string, int> Relu::get_params(
) {
    std::unordered_map<std::string, int> param_dict;
    param_dict["N"] = N;
    param_dict["C"] = C;
    param_dict["H"] = H;
    param_dict["W"] = W;
    return param_dict;
}

/**
 * @brief Relu forward_pass
 * 
 * @param in input from previous layer's forward pass
 * @return d_type* current relu layer's output
 */
void Relu::forward_pass(
    d_type *in,
    d_type *out
) {
    // Relu::in = in;

    const int num = N*C*H*W;
    int i=0;

    while (i < num) {
        out[i] = in[i] > 0 ? in[i] : 0;
        ++i;
    }
}

/**
 * @brief Relu backward_pass
 * 
 * @param dout input from previous layer's backward pass
 * @return std::unordered_map<std::string, d_type*> containing dx for gradient descent
 */
std::unordered_map<std::string, d_type*> Relu::backward_pass(
    d_type *dout
) {
    const int num = N*C*H*W;
    int i=0;

    while (i < num) {
        dx[i] = in[i] > 0 ? dout[i] : 0;
        ++i;
    }

    std::unordered_map<std::string, d_type*> res_dict;
    res_dict["dx"] = dx;

    return res_dict;
}
