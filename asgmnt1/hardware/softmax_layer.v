`timescale 1ns / 1ps

// this module is pure combinational, so clk, rst_n and done signals are not needed
module softmax_layer #(parameter BUS_WIDTH = 32,
                       parameter NUM_DECIMAL_IN_BINARY = 6) (
        input logic signed [BUS_WIDTH-1:0] dense_sum2[10],
        output logic signed [9:0] dense_softmax[10]
    );
    
    logic signed [25:0] exp_dense_sum2[10];
    logic signed [25:0] den;
    
    assign den = exp_dense_sum2[0] + exp_dense_sum2[1] + exp_dense_sum2[2] + exp_dense_sum2[3] + exp_dense_sum2[4] + 
                 exp_dense_sum2[5] + exp_dense_sum2[6] + exp_dense_sum2[7] + exp_dense_sum2[8] + exp_dense_sum2[9];
    
    genvar i;
    generate
        for (i=0; i<10; i++) begin
            hw_exp #(
                .BUS_WIDTH(BUS_WIDTH),
                .NUM_DECIMAL_IN_BINARY(NUM_DECIMAL_IN_BINARY)
            ) exp (
                .x(dense_sum2[i]),
                .ans(exp_dense_sum2[i])
            );
            
            fixed_point_div #(
                .BUS_WIDTH(26),
                .NUM_DECIMAL_IN_BINARY(NUM_DECIMAL_IN_BINARY)
            ) div (
                .a(exp_dense_sum2[i]),
                .b(den),
                .ans(dense_softmax[i])
            );
        end
    endgenerate
    
endmodule
