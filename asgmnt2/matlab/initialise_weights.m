function [conv_w, conv_b, dense_w, dense_b, time] = initialise_weights(data_type)
    tic;
    conv_w = -1 + (1+1)*rand(5,7,7,data_type);
    conv_b = -1 + (1+1)*rand(5,28,28,data_type);
    dense_w = -1 + (1+1)*rand(980,10,data_type);
    dense_b = -1 + (1+1)*rand(1,10,data_type);
    time = toc;
end
