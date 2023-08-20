function output = convolusion_bfloat16(inputs, weights, bias)
    data_type = "bfloat16";
    inputs = my_cast(inputs, data_type);
    weights = my_cast(weights, data_type);
    bias = my_cast(bias, data_type);

    output = bias + dot(inputs, weights);
    % output = bias + sum(inputs .* weights);
end
