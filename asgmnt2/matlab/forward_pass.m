function [max_pooling, dense_input, dense_softmax, time] = forward_pass(img, ...
    conv_w, conv_b, dense_w, dense_b, data_type)

    tic;

    max_pooling = zeros([5,28,28],data_type);
    dense_input = zeros([1,980],data_type);
    dense_sum = zeros([1,120],data_type);
    dense_softmax = zeros([1,10],data_type);
    
    conv_layer = zeros([5,28,28],data_type);
    max_layer = zeros([5,14,14],data_type);
    
    temp_img = reshape(transpose(img), [1, length(img(1,:))*length(img)]);
    
    % Convolution Operation + Sigmoid Activation
    parfor filter_dim=1:5
        for i=1:28
            for j=1:28
                temp_wights = zeros([1,7*7],data_type);
                temp_inputs = zeros([1,7*7],data_type);
                for k=1:7
                    for l=1:7
                        temp_wights((k-1)*7 + l) = cast(temp_img((i + k + 1 - 1 - 1)*32 + j + l - 2 - 1), data_type);
                        temp_inputs((k-1)*7 + l) = conv_w(filter_dim, k, l);
                    end
                end
                conv_layer(filter_dim, i, j) = convolusion(temp_inputs, temp_wights, conv_b(filter_dim, i, j));
            end
        end
    end
    sig_layer = 1 ./ (1 + exp(-1 .* conv_layer));
%     sig_layer = logsig(conv_layer);
    
    % MAX Pooling (max_pooling, max_layer)
    for filter_dim=1:5
        for i=1:2:28
            for j=1:2:28
                matrix = sig_layer(filter_dim,i:i+1,j:j+1);
                [M,I] = max(matrix);
                [M2,I2] = max(M);
                I = [I(I2) I2];
                max_pooling(filter_dim, I(1), I(2)) = 1;
                max_layer(filter_dim, (i+1)/2, (j+1)/2) = M2;
            end
        end
    end
    for i=1:5
        for j=1:14
            dense_input((i-1)*14*14 + (j-1)*14 + 1 : (i-1)*14*14 + (j-1)*14 + 14) = reshape(max_layer(i,j,:), [1 14]);
        end
    end
    
    % Dense Layer
    for i=1:10
        dense_sum(i) = convolusion(transpose(dense_w(:,i)), dense_input, dense_b(i));
    end
    
    % Softmax Output
    dense_softmax_new = exp(dense_sum);
    for i=1:10
        dense_softmax(i) = dense_softmax_new(i) / sum(dense_softmax_new);
    end

    time = toc;
end
