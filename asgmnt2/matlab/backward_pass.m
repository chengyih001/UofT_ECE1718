function [db1, dw1, dw_conv, db_conv, dw_max, time] = backward_pass(dense_softmax, y, img, dense_w, ...
    max_pooling, dense_input, data_type)
    
    tic;

    delta4 = dense_softmax - cast(y,data_type); % Derivative of Softmax + Cross entropy
    db1 = delta4; % Bias Changes

    % Calculate Weight Changes for Dense Layer 2
    dw1 = zeros([980,10],data_type);
    for i=1:980
        dw1(i,:) = dense_input(i) * delta4;
    end

    % Delta 3
    delta3 = zeros([1,980],data_type);
    for i=1:980
        delta3(i) = convolution(dense_w(i,:), delta4, 0);
        %delta3(i) = sum(dense_w(i,:) .* delta4);
    end
    delta3 = delta3 .* d_sigmoid(dense_input);

    % Calc back-propagated max layer dw_max
    dw_max = zeros([5,28,28],data_type);
    k = 1;
    for filter_dim=1:5
        for i=1:2:28
            for j=1:2:28
                if ( sum(sum(max_pooling(filter_dim, i:i+1, j:j+1))) >= 1)
                    dw_max(filter_dim,i,j) = delta3(k);
                else
                    dw_max(filter_dim,i,j) = 0;
                end
                k = k + 1;
            end
        end
    end

    % Calc Conv Bias Changes
    db_conv = dw_max;

    % Set Conv Layer Weight changes to 0
    dw_conv = zeros([5,7,7],data_type);

    % Calculate Weight Changes for Conv Layer
    temp_img = reshape(transpose(img), [1, length(img(1,:))*length(img)]);
    for filter_dim=1:5
        for i=1:2:28
            for j=1:2:28
                cur_val = dw_max(filter_dim, i, j);
                for k=1:7
                    for l=1:7
                        dw_conv(filter_dim, k, l) = dw_conv(filter_dim, k, l) + cast(temp_img((i + k + 1 - 1 - 1)*32 + j + l - 2 - 1), data_type) * cur_val;
                    end
                end
            end
        end
    end

    time = toc;
end
