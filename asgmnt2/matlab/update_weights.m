function [conv_w, conv_b, dense_w, dense_b, time] = update_weights(db1, ...
    dw1, dw_conv, db_conv, conv_w, conv_b, dense_w, dense_b)
    
    tic;

    dense_b = dense_b - 0.01 * db1;
    for i=1:10
        dense_w(:,i) = dense_w(:,i) - 0.01 * dw1(:,i);
    end

    conv_w = conv_w - 0.01 * dw_conv;
    conv_b = conv_b - 0.01 * db_conv;

    time = toc;
end