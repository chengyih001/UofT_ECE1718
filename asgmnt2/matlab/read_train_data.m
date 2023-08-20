function [label_train,data_train, time] = read_train_data()
    tic;
    table = readtable('C:/Course Material/UofT/ECE1718/assignment2/mnist_train.csv');
    table = table{:,:};
    table = cast(table, 'uint8');
    label_train = table(:,1).';
    data_train = table(:, 2:length(table(1,:)));
    time = toc;
end