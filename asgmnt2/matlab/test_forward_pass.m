clear
clc

data_type = "single";
corr = 0;
test_len = 20;

load('weights_pruned.mat');
conv_w = matrix_3d_to_2d(conv_w, data_type);
conv_b = matrix_3d_to_2d(conv_b, data_type);
dense_w = single(dense_w);
dense_b = single(dense_b);

load('test_set.mat');

for num_img = 1:test_len
    img = give_img(data_test(num_img,:));

    temp_img = reshape(transpose(img), [1, length(img(1,:))*length(img)]);
    
    [~, ~, dense_softmax] = forward_pass_2d(temp_img, conv_w, conv_b, dense_w, dense_b);
    
    % analyze the result
    dense_softmax;
    [~,max_pos] = max(dense_softmax);
    act = max_pos - 1;
    exp = label_test(num_img);
    if (act == exp)
        corr = corr+1;
    end
%     if (mod(num_img, 10) == 0)
        fprintf("done testing image %d\n", num_img);
%     end
    clear exp
    clear act
end

% analyze the total result
fprintf("Accuracy: %f\n", corr/test_len);
