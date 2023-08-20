function [label_test,data_test, time] = read_test_data()
    tic;
    table = readtable('C:/Course Material/UofT/ECE1718/assignment2/mnist_test.csv');
    table = table{:,:};
    table = cast(table, 'uint8');
    label_test = table(:,1).';
    data_test = table(:, 2:length(table(1,:)));
    save('test_set.mat');
    time = toc;
end