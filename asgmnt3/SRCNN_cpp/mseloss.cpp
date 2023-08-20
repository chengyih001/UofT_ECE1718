#include "mseloss.h"

mseloss::mseloss(
) {
    loss = 0.0;
    out = nullptr;
}

mseloss::~mseloss(
) {
    free(out);
}

void mseloss::set_params(
    std::unordered_map<std::string, int> param_dict
) {
    N = param_dict["N"];
    C = param_dict["C"];
    H = param_dict["H"];
    W = param_dict["W"];

    out = (d_type *)calloc(N*C*H*W, sizeof(d_type));
}

double mseloss::forward_pass(
    d_type *y_pred,
    d_type *y_label
) {
    const int num = N*C*H*W;
    int i=0;

    while (i < num) {
        loss += pow((y_pred[i] - y_label[i]), 2.0);
        ++i;
    }

    loss /= 2*double(3*32*32);

    return loss;
}

d_type* mseloss::backward_pass(
    d_type *y_pred,
    d_type *y_label
) {
    const int num = N*C*H*W;
    int i=0;

    while (i < num) {
        out[i] = (y_pred[i] - y_label[i]) / double(3*32*32);
    }

    return out;
}