#ifndef RELU_H
#define RELU_H

#include <unordered_map>
#include <string>

typedef float d_type;

/**
 * x: (N, C, H, W)
 * 
 */
class Relu {
    private:
        int N, C, H, W;

        d_type *in;

    public:
        d_type *dx;

        Relu();
        Relu(std::unordered_map<std::string, int>);
        ~Relu();

        void set_params(std::unordered_map<std::string, int>);
        std::unordered_map<std::string, int> get_params();

        void forward_pass(d_type*, d_type*);
        std::unordered_map<std::string, d_type*> backward_pass(d_type*);
};

#endif