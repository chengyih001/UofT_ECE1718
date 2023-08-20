#ifndef MSELOSS_H
#define MSELOSS_H

#include <unordered_map>
#include <string>

typedef float d_type;

class mseloss {
    private:
        int N, C, H, W;

    public:
        d_type *out;
        double loss;

        mseloss();
        ~mseloss();

        void set_params(std::unordered_map<std::string, int>);

        double forward_pass(d_type*, d_type *);
        d_type *backward_pass(d_type*, d_type *);
};

#endif