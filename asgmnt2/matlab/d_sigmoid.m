function output = d_sigmoid(x)
%     sig = logsig(x);
    sig = 1 ./ (1 + exp(-1 .* x));
    output = sig .* (1 - sig);
end