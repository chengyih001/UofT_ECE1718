#ifndef SRCNN_H
#define SRCNN_H

#include <iostream>
using namespace std;

#include "conv.h"
#include "relu.h"
#include <unordered_map>
#include <string>
#include <math.h>

#define N 1

typedef float d_type;

class SRCNN {
    private:
        Conv2d conv1, conv2, conv3;
        Relu relu1, relu2;

    public:
        double loss;

        SRCNN();
        ~SRCNN();

        void set_params(
            d_type*, d_type*,
            d_type*, d_type*,
            d_type*, d_type*
        );

        void forward_pass(d_type*, d_type*);
        std::unordered_map<std::string, std::unordered_map<std::string, d_type*> > backward_pass(d_type*);
};

#endif