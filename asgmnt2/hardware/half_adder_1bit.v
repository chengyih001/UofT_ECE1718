`timescale 1ns / 1ps

module half_adder_1bit(a, b, sum, c_out);
    input a;
    input b;
    output sum;
    output c_out;
    
    assign sum = a ^ b;
    assign c_out = a & b;
    
endmodule
