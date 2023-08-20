clear
clc

conv_w = zeros(5, 7, 7, 'double');
conv_b = zeros(5, 28, 28, 'double');
dense_w = zeros(980, 10, 'double');
dense_b = zeros(1, 10, 'double');

weights = fopen('weights_pruned.txt');

for i = 1:5
    for j = 1:7
        for k = 1:7
            tline = fgetl(weights);
            [temp, value]= strread(tline,'%s = %f');
            conv_w(i,j,k) = value;
        end
    end
end

for i = 1:5
    for j = 1:28
        for k = 1:28
            tline = fgetl(weights);
            [temp, value]= strread(tline,'%s = %f');
            conv_b(i,j,k) = value;
        end
    end
end

for i = 1:980
    for j = 1:10
        tline = fgetl(weights);
        [temp, value]= strread(tline,'%s = %f');
        dense_w(i,j) = value;
    end
end

for i = 1:10
    tline = fgetl(weights);
    [temp, value]= strread(tline,'%s = %f');
    dense_b(i) = value;
end

fclose(weights);
save('weights_pruned.mat');