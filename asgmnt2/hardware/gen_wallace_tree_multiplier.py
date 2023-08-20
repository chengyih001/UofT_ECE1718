# python code for generate wallace_tree_multiplier Verilog code

import math

def get_num_num_zero_in_col(in_list, col_num, exclude_string):
    count = 0
    first_index = -1
    for i in range(len(in_list)):
        if (in_list[i][col_num] != "" and in_list[i][col_num] != 0 and exclude_string not in in_list[i][col_num]):
            if (first_index == -1):
                first_index = i
            count = count + 1
    return (first_index, count)

def shift_col_down(in_list, col):
    temp_list = []
    for i in range(len(in_list)):
        if (in_list[i][col] != "" and in_list[i][col] != 0):
            temp_list.append(in_list[i][col])
            in_list[i][col] = ""
    for i in range(1, len(temp_list)+1):
        in_list[-i][col] = temp_list[-i]

def gen_partial_products(op_size):
    # initialize partial_products list
    partial_products = []
    for i in range(op_size):
        partial_products.append([])
        count = op_size-1
        partial_products[i].append("")
        for j in range(op_size*2-1):
            if (j >= op_size-1-i and j <= op_size*2-2-i):
                partial_products[i].append("par_product[{}][{}]".format(i, count))
                count = count -1
            else:
                partial_products[i].append("")
    # shift the list
    for i in range(op_size*2):
        while (partial_products[-1][i] == "" and i != 0):
            for j in range(1, op_size):
                partial_products[-j][i] = partial_products[-j-1][i]
            partial_products[0][i] = ""
    return partial_products

def add_half_adder(partial_products, col, index, list_carry_to_add, stage):
    # initialize the half adder
    adder = {
        "adder name" : "half_adder_stage{}_{}".format(stage, len(list_adder[stage])),
        "adder type" : "half_adder_1bit",
        "port a" : partial_products[index][col],
        "port b" : partial_products[index+1][col],
        "port sum" : "half_adder_stage{}_{}_sum".format(stage, len(list_adder[stage])),
        "port c_out" : "half_adder_stage{}_{}_c_out".format(stage, len(list_adder[stage])),
    }
    list_adder[stage].append(adder)
    # do the compression
    partial_products[index][col] = ""
    partial_products[index+1][col] = adder["port sum"]
    # add the carry to the 'list_carry_to_add' which will goes to the next column in the next stage
    list_carry_to_add[col-1].append(adder["port c_out"])
    return

def add_full_adder(partial_products, col, index, list_carry_to_add, stage):
    # initialize the half adder
    adder = {
        "adder name" : "full_adder_stage{}_{}".format(stage, len(list_adder[stage])),
        "adder type" : "full_adder_1bit",
        "port a" : partial_products[index][col],
        "port b" : partial_products[index+1][col],
        "port c_in" : partial_products[index+2][col],
        "port sum" : "full_adder_stage{}_{}_sum".format(stage, len(list_adder[stage])),
        "port c_out" : "full_adder_stage{}_{}_c_out".format(stage, len(list_adder[stage])),
    }
    list_adder[stage].append(adder)
    # do the compression
    partial_products[index][col] = ""
    partial_products[index+1][col] = ""
    partial_products[index+2][col] = adder["port sum"]
    # add the carry to the 'list_carry_to_add' which will goes to the next column in the next stage
    list_carry_to_add[col-1].append(adder["port c_out"])
    return

def compress_partial_products(partial_products):
    stage = -1
    stop = False
    
    # do one stage of compression
    while (stop == False):
        list_carry_to_add = []
        for i in range(len(partial_products[0])):
            list_carry_to_add.append([])
        
        beginning = True
        stage = stage + 1
        list_adder.append([])
        for i in range(len(partial_products[0])):
            col = len(partial_products[0]) - i - 1
            (first_index, non_zero_count) = get_num_num_zero_in_col(partial_products, col, " ")
            # if (non_zero_count > 2 and beginning == False):
            while (non_zero_count >= 2 and beginning == False):
                # adders can be used to compress the partial products
                if (non_zero_count >= 3):
                    add_full_adder(partial_products, col, first_index, list_carry_to_add, stage)
                else:
                    add_half_adder(partial_products, col, first_index, list_carry_to_add, stage)
                (first_index, non_zero_count) = get_num_num_zero_in_col(partial_products, col, "stage{}".format(stage))
            if (non_zero_count >= 3 and beginning == True):
                beginning = False
                # adders can be used to compress the partial products
                add_half_adder(partial_products, col, first_index, list_carry_to_add, stage)
            shift_col_down(partial_products, col)
        # add carry from last stage to the partial products
        if (list_carry_to_add[-1] != []):
            # should be fine since multipier's output only takes (2*input bus width) bits
            print("Warning! carry overflowed! Forcing the list_carry_to_add[-1] to empty!")
            list_carry_to_add[-1] = []
        for i in range(1, len(partial_products[0])+1):
            (first_index, non_zero_count) = get_num_num_zero_in_col(partial_products, -i, " ")
            if (first_index == -1): # incase of non-zero value not found
                first_index = 0
            for j in range(1, len(list_carry_to_add[-i])+1):
                partial_products[first_index-j][-i] = list_carry_to_add[-i][-j]
        # check if this is the time to stop
        stop = True
        for i in range(len(partial_products[0])):
            (first_index, non_zero_count) = get_num_num_zero_in_col(partial_products, i, " ")
            if (non_zero_count > 2):
                stop = False
                break

def code_gen(num_bits, list_adder, partial_products):
    code = '''// this code is auto-generated by the python script gen_wallace_tree_multiplier.py\n
`timescale 1ns / 1ps

module wallace_tree_multiplier_{}bits (
    input logic signed [{}:0] a,
    input logic signed [{}:0] b,
    output logic signed [{}:0] product
);

'''.format(num_bits, num_bits-1, num_bits-1, num_bits*2-1)
    code = code + "    // get absolute value of a and b\n"
    code = code + "    logic [{}:0] pos_a, pos_b;\n".format(num_bits-1)
    code = code + "    assign pos_a = (a[{}] == 1'b1) ? (({}'h{} ^ a) + {}'b1) : (a);\n".format(num_bits-1, num_bits, "f"*math.ceil(num_bits/4), num_bits)
    code = code + "    assign pos_b = (b[{}] == 1'b1) ? (({}'h{} ^ b) + {}'b1) : (b);\n\n".format(num_bits-1, num_bits, "f"*math.ceil(num_bits/4), num_bits)

    code = code + "    // get partial product by a[i] AND b\n"
    code = code + "    logic [{}:0] par_product [{}:0];".format(num_bits-1, num_bits-1)
    code = code + '''
    genvar i;
    generate
        for (i=0; i<{}; i++) begin
'''.format(num_bits)
    code = code + "            assign par_product[i] = {(" + str(num_bits) + "){pos_a[i]}} & pos_b;\n"
    code = code + "        end\n"
    code = code + "    endgenerate\n\n"

    # put all the adder instantiations
    stage = 0
    code = code + "    // adders for different stages of compression\n"
    for adder_list in list_adder:
        code = code + "    // adders in stage {}\n".format(stage)
        for adder in adder_list:
            code = code + "    logic " + adder["port sum"] + "; \n"
            code = code + "    logic " + adder["port c_out"] + "; \n"
            code = code + "    " + adder["adder type"] + " " + adder["adder name"] + " (\n"
            code = code + "        .a(" + adder["port a"] + "), \n"
            code = code + "        .b(" + adder["port b"] + "), \n"
            if (adder["adder type"] == "full_adder_1bit"):
                code = code + "        .c_in(" + adder["port c_in"] + "), \n"
            code = code + "        .sum(" + adder["port sum"] + "), \n"
            code = code + "        .c_out(" + adder["port c_out"] + ") \n"
            code = code + "    ); \n"
        code = code + "\n"
        stage = stage + 1
    
    # put get the final result from the partial_products
    code = code + "    // adders for outputs\n"
    code = code + "    logic [{}:0] pos_product;\n".format(num_bits*2-1)
    code = code + "    logic output_adder_c_out [{}:0]; \n".format(num_bits*2-1)
    for i in range(len(partial_products[0])):
        bit_index = 2*num_bits -1 - i
        if (partial_products[-1][bit_index] != "" and partial_products[-2][bit_index] != ""):
            code = code + "    full_adder_1bit " + "output_adder_{}".format(i) + " (\n"
            code = code + "        .a(" + partial_products[-1][bit_index] + "), \n"
            code = code + "        .b(" + partial_products[-2][bit_index] + "), \n"
            code = code + "        .c_in(output_adder_c_out[" + str(i-1) + "]), \n"
            code = code + "        .sum(pos_product[" + str(i) + "]), \n"
            code = code + "        .c_out(output_adder_c_out[" + str(i) + "]) \n"
            code = code + "    );\n"
        else:
            code = code + "    assign pos_product[" + str(i) + "] = " + partial_products[-1][bit_index] + ";\n"
            code = code + "    assign output_adder_c_out[" + str(i) + "] = 1'b0; \n"

    code = code + "\n    // get product from pos_product\n"
    code = code + "    logic product_sign;\n"
    code = code + "    assign product_sign = a[{}] ^ b[{}];\n".format(num_bits-1, num_bits-1)
    code = code + "    assign product = (product_sign == 1'b1) ? (({}'h{} ^ pos_product) + {}'b1) : (pos_product);\n\n".format(num_bits*2, "f"*math.ceil(num_bits*2/4), num_bits*2)

    # closure
    code = code + "endmodule"
    return code


# main 
num_bits = 32
file_name = "wallace_tree_multiplier.v".format(num_bits)
list_adder = []

# simulate the compression to know the connections of adders
partial_products = gen_partial_products(num_bits)
compress_partial_products(partial_products)

# generate the code based on the simulated result
code = code_gen(num_bits, list_adder, partial_products)

# save the code into the file
f = open(file_name, "w")
f.write(code)
f.close()
print("Code generation finished. '" + file_name + "' has been generated. ")
