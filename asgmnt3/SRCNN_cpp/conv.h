#ifndef CONV_H
#define CONV_H

#include <iostream>
using namespace std;

#include <unordered_map>
#include <string>

#define DEFAULT_STRIDE 1
#define DEFAULT_PAD 0

typedef float d_type;

/**
 * x: (N, C, H_in, W_in)
 * w: (F, C, H_w, W_w)
 * b: (F,)
 * 
 * out: (N, F, H_out, W_out)
 *      H_out = 1 + (H_in + 2 * padding - H_w) / stride
 *      W_out = 1 + (W_in + 2 * padding - W_w) / stride
 * 
 * cache: (x_padded, w, b, stride, padding)
 */
class Conv2d {
    private:
        int N;
        int C, H_in, W_in;
        int F, H_w, W_w;
        int H_out, W_out;
        int stride, padding;

        d_type *in_padded, *weight, *bias;

        void _padding(d_type*);
    public:
        d_type *dx, *dw, *db;

        Conv2d();
        Conv2d(std::unordered_map<std::string, int>, d_type*, d_type*);
        ~Conv2d();

        void set_params(std::unordered_map<std::string, int>, d_type*, d_type*);
        std::unordered_map<std::string, int> get_params();

        void forward_pass(d_type*, d_type*);
        std::unordered_map<std::string, d_type*> backward_pass(d_type*);
};

#endif