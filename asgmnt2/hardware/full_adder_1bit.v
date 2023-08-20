`timescale 1ns / 1ps

module full_adder_1bit(a, b, c_in, sum, c_out);

    input a;
    input b;
    input c_in;
    output sum;
    output c_out;
    
    assign sum = (a ^ b) ^ c_in;
    assign c_out = ((a ^ b) & c_in) | (a & b);
    
endmodule
