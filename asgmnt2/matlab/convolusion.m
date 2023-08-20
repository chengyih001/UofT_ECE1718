function output = convolusion(inputs, weights, bias)
    output = bias + sum(inputs .* weights);
end