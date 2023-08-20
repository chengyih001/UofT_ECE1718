function output = matrix_3d_to_2d(input, data_type)
    matrix_size = size(input);
    output = zeros([matrix_size(1), matrix_size(2) * matrix_size(3)], data_type);
    for i=1:matrix_size(1)
        for j=1:matrix_size(2)
            output(i, (j-1)*matrix_size(3) + 1 : (j)*matrix_size(3) ) = input(i,j,:);
        end
    end
end