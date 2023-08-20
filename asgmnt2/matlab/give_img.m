function img = give_img(vec)
    img = zeros(35,32,'uint8');
    k=1;
    for i=1:35
        for j=1:32
            if (i < 6|| j < 3 || i > 33 || j > 30)
                img(i,j) = 0;
            else
                img(i,j) = vec(k);
                k = k + 1;
            end
        end
    end
end