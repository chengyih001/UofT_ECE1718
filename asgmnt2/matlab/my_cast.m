function [output] = my_cast(input, data_type)
    if (data_type ~= "bfloat16")
        output = cast(input, data_type);
    else
        output = CustomFloat(input, 16, 7);
    end
end