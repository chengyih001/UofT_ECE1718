#include <hls_stream.h>
#include <hls_vector.h>
#include <cmath>
#include <stdio.h>

typedef float d_type;

d_type sigmoid(d_type x);

d_type softmax_den(d_type *x, int len);

extern "C"

void vadd(
            d_type* in, d_type* out,
            d_type* conv_w, d_type* conv_b,
            d_type* dense1_w, d_type* dense1_b,
            d_type* dense2_w, d_type* dense2_b
        );

