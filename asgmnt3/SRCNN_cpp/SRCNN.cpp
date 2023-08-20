#include "SRCNN.h"

SRCNN::SRCNN(
) {
    loss = 0;
}

SRCNN::~SRCNN(
) {
    // conv1.~Conv2d();
    // conv2.~Conv2d();
    // conv3.~Conv2d();
    // relu1.~Relu();
    // relu2.~Relu();
}

void SRCNN::set_params(
    d_type* conv1_weight, d_type* conv1_bias,
    d_type* conv2_weight, d_type* conv2_bias,
    d_type* conv3_weight, d_type* conv3_bias
) {
    std::unordered_map<std::string, int> conv1_param_dict;
    std::unordered_map<std::string, int> relu1_param_dict;
    std::unordered_map<std::string, int> conv2_param_dict;
    std::unordered_map<std::string, int> relu2_param_dict;
    std::unordered_map<std::string, int> conv3_param_dict;

    conv1_param_dict["N"] = N;
    conv1_param_dict["C"] = 3;
    conv1_param_dict["H_in"] = 32;
    conv1_param_dict["W_in"] = 32;
    conv1_param_dict["F"] = 64;
    conv1_param_dict["H_w"] = 9; 
    conv1_param_dict["W_w"] = 9;
    conv1_param_dict["STRIDE"] = 1;
    conv1_param_dict["PADDING"] = 2;

    conv2_param_dict["N"] = N;
    conv2_param_dict["C"] = 64;
    conv2_param_dict["H_in"] = 1 + (conv1_param_dict["H_in"] + 2 * conv1_param_dict["PADDING"] - conv1_param_dict["H_w"]) / conv1_param_dict["STRIDE"];
    conv2_param_dict["W_in"] =  1 + (conv1_param_dict["W_in"] + 2 * conv1_param_dict["PADDING"] - conv1_param_dict["W_w"]) / conv1_param_dict["STRIDE"];
    conv2_param_dict["F"] = 32;
    conv2_param_dict["H_w"] = 1; 
    conv2_param_dict["W_w"] = 1;
    conv2_param_dict["STRIDE"] = 1;
    conv2_param_dict["PADDING"] = 2;
    
    conv3_param_dict["N"] = N;
    conv3_param_dict["C"] = 32;
    conv3_param_dict["H_in"] = 1 + (conv2_param_dict["H_in"] + 2 * conv2_param_dict["PADDING"] - conv2_param_dict["H_w"]) / conv2_param_dict["STRIDE"];
    conv3_param_dict["W_in"] =  1 + (conv2_param_dict["H_in"] + 2 * conv2_param_dict["PADDING"] - conv2_param_dict["H_w"]) / conv2_param_dict["STRIDE"];
    conv3_param_dict["F"] = 3;
    conv3_param_dict["H_w"] = 5; 
    conv3_param_dict["W_w"] = 5;
    conv3_param_dict["STRIDE"] = 1;
    conv3_param_dict["PADDING"] = 2;

    relu1_param_dict["N"] = N;
    relu1_param_dict["C"] = 64;
    relu1_param_dict["H"] = 1 + (conv1_param_dict["H_in"] + 2 * conv1_param_dict["PADDING"] - conv1_param_dict["H_w"]) / conv1_param_dict["STRIDE"];
    relu1_param_dict["W"] = 1 + (conv1_param_dict["W_in"] + 2 * conv1_param_dict["PADDING"] - conv1_param_dict["W_w"]) / conv1_param_dict["STRIDE"];;

    relu2_param_dict["N"] = N;
    relu2_param_dict["C"] = 32;
    relu2_param_dict["H"] = 1 + (conv2_param_dict["H_in"] + 2 * conv2_param_dict["PADDING"] - conv2_param_dict["H_w"]) / conv2_param_dict["STRIDE"];
    relu2_param_dict["W"] = 1 + (conv2_param_dict["H_in"] + 2 * conv2_param_dict["PADDING"] - conv2_param_dict["H_w"]) / conv2_param_dict["STRIDE"];

    conv1.set_params(conv1_param_dict, conv1_weight, conv1_bias);
    relu1.set_params(relu1_param_dict);
    conv2.set_params(conv2_param_dict, conv2_weight, conv2_bias);
    relu2.set_params(relu2_param_dict);
    conv3.set_params(conv3_param_dict, conv3_weight, conv3_bias);
}

void SRCNN::forward_pass(
    d_type *img,
    d_type *out
) {
    d_type *temp_out = (d_type *)calloc(1*64*28*28, sizeof(d_type));;

    conv1.forward_pass(img, temp_out);
    relu1.forward_pass(temp_out, temp_out);

    conv2.forward_pass(temp_out, temp_out);
    relu2.forward_pass(temp_out, temp_out);

    conv3.forward_pass(temp_out, out);
    
    // free(temp_out);
}

std::unordered_map<std::string, std::unordered_map<std::string, d_type*> > SRCNN::backward_pass(
    d_type *dout
) {
    d_type *temp_dout;

    std::unordered_map<std::string, d_type*> conv3_grad_dict;
    std::unordered_map<std::string, d_type*> relu2_grad_dict;
    std::unordered_map<std::string, d_type*> conv2_grad_dict;
    std::unordered_map<std::string, d_type*> relu1_grad_dict;
    std::unordered_map<std::string, d_type*> conv1_grad_dict;

    conv3_grad_dict = conv3.backward_pass(dout);
    relu2_grad_dict = relu2.backward_pass(conv3_grad_dict["dx"]);
    conv2_grad_dict = conv2.backward_pass(relu2_grad_dict["dx"]);
    relu1_grad_dict = relu1.backward_pass(conv2_grad_dict["dx"]);
    conv1_grad_dict = conv1.backward_pass(relu1_grad_dict["dx"]);

    std::unordered_map<std::string, std::unordered_map<std::string, d_type*> > SRCNN_grad_dict;
    SRCNN_grad_dict["conv3_dict"] = conv3_grad_dict;
    SRCNN_grad_dict["relu2_dict"] = relu2_grad_dict;
    SRCNN_grad_dict["conv2_dict"] = conv2_grad_dict;
    SRCNN_grad_dict["relu1_dict"] = relu1_grad_dict;
    SRCNN_grad_dict["conv1_dict"] = conv1_grad_dict;

    return SRCNN_grad_dict;
}
