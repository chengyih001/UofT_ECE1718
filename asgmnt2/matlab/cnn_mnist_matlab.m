clear
clc

data_type = "double";

time_forward_pass = 0;
time_backward_pass = 0;
time_update_weights = 0;

epoch = 500;
batch_size = 200;
test_len = 600;

fprintf("Loaindg data.\n")
[label_test,data_test, time_read_test_data] = read_test_data();
[label_train,data_train, time_read_train_data] = read_train_data();
[conv_w, conv_b, dense_w, dense_b, time_initialise_weights] = initialise_weights(data_type);

fprintf("Start Training.\n");
for i=1:epoch
    for j=1:batch_size
        num = randi([1 60000]);
        vector_y = give_y(label_train(num));
        img = give_img(data_train(num,:));
        
        [max_pooling, dense_input, dense_softmax, time] = forward_pass(img, conv_w, ...
            conv_b, dense_w, dense_b, data_type);
        time_forward_pass = time_forward_pass + time;

        [db1, dw1, dw_conv, db_conv, dw_max, time] = backward_pass(dense_softmax, vector_y, img, dense_w, ...
            max_pooling, dense_input, data_type);
        time_backward_pass = time_backward_pass + time;

        [conv_w, conv_b, dense_w, dense_b, time] = update_weights(db1, ...
            dw1, dw_conv, db_conv, conv_w, conv_b, dense_w, dense_b);
        time_update_weights = time_update_weights + time;
    end
    fprintf("Epoch %d done..\n", i);
end

cor = 0;
fprintf("Start Testing.\n");
for i=1:test_len
    num = i;
    img = give_img(data_test(num,:));

    [~, ~, dense_softmax, time] = forward_pass(img, conv_w, conv_b, dense_w, dense_b, data_type);
    time_forward_pass = time_forward_pass + time;

    [~,max_pos] = max(dense_softmax);
    act = max_pos - 1;
    exp = label_test(num);
    if (act == exp)
        cor = cor+1;
    end
    if (mod(num, 10) == 0)
        fprintf("done testing image %d\n", num);
    end
    clear exp
    clear act
end

% analyze the total result
fprintf("Accuracy: %f\n", cor/test_len);
fprintf("Time spend for functions: \n");
fprintf("   read_test_data : %f \n", time_read_test_data);
fprintf("   read_train_data : %f \n", time_read_train_data);
fprintf("   initialise_weights : %f \n", time_initialise_weights);
fprintf("   forward_pass : %f \n", time_forward_pass);
fprintf("   backward_pass : %f \n", time_backward_pass);
fprintf("   update_weights : %f \n", time_update_weights);
