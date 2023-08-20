function [max_pooling, dense_input, dense_softmax] = forward_pass_2d(img, ...
    conv_w, conv_b, dense_w, dense_b)

    max_pooling = single(zeros([5,28*28]));
    dense_input = single(zeros([1,980]));
    dense_sum = single(zeros([1,10]));
    dense_softmax_new = single(zeros([1,10]));
    dense_softmax = single(zeros([1,10]));
    
    sig_layer = single(zeros([5,28*28]));
    conv_layer = single(zeros([5,28*28]));
    max_layer = single(zeros([5,14*14]));
    
    % Convolution Operation + Sigmoid Activation
    for filter_dim=uint32(1:5)
        for i=uint32(1:28)
            for j=uint32(1:28)
                temp_wights = single(zeros([1,7*7]));
                temp_inputs = single(zeros([1,7*7]));
                for k=uint32(1:7)
                    for l=uint32(1:7)
                        temp_wights((k-1)*7 + l) = single(img((i + k + 1 - 1 - 1)*32 + j + l - 2 - 1));
                        temp_inputs((k-1)*7 + l) = conv_w(filter_dim, (k-1)*7 + l);
                    end
                end
                conv_layer(filter_dim, (i-1)*28+j) = convolusion(temp_inputs, temp_wights, conv_b(filter_dim, (i-1)*28+j));
            end
        end
    end
    for filter_dim=uint32(1:5)
        for i=uint32(1:28*28)
            temp_val = -1 * conv_layer(filter_dim, i);
            sig_layer(filter_dim, i) = 1 / (1 + exp_hw(temp_val));
        end
    end
%     sig_layer = 1 ./ (1 + exp(-1 .* conv_layer));
%     sig_layer = logsig(conv_layer);
    
    % MAX Pooling (max_pooling, max_layer)
    for filter_dim=uint32(1:5)
        for i=uint32(1:2:28)
            for j=uint32(1:2:28)
                matrix = [sig_layer(filter_dim, (i-1)*28+j), sig_layer(filter_dim, (i-1)*28+j+1), sig_layer(filter_dim, (i)*28+j), sig_layer(filter_dim, (i)*28+j+1)];
                M = max(matrix);
                if (M == matrix(1))
                    I = single(1);
                elseif (M == matrix(2))
                    I = single(2);
                elseif (M == matrix(3))
                    I = single(3);
                else
                    I = single(4);
                end
                max_i = uint32(single(i) + ceil(I/2) - 1);
                max_j = uint32(single(j) + 2-mod(I,2) - 1);
                max_pooling(filter_dim, (max_i-1)*28+max_j) = 1;
                max_layer(filter_dim, ((i+1)/2-1)*14+(j+1)/2) = M;
            end
        end
    end
    for i=uint32(1:5)
        dense_input((i-1)*14*14 + 1 : (i-1)*14*14 + 14*14) = reshape(max_layer(i,:), [1 14*14]);
    end
    
    % Dense Layer
    for i=uint32(1:10)
        dense_sum(i) = convolusion(transpose(dense_w(:,i)), dense_input, dense_b(i));
        dense_softmax_new(i) = exp_hw(dense_sum(i));
    end
    
    % Softmax Output
%     dense_softmax_new = exp(dense_sum);
    for i=uint32(1:10)
        dense_softmax(i) = dense_softmax_new(i) / sum(dense_softmax_new);
    end
end