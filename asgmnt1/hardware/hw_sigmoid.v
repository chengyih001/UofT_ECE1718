`timescale 1ns / 1ps

module hw_sigmoid #(parameter BUS_WIDTH = 32,
                    parameter NUM_DECIMAL_IN_BINARY = 6) (
                    input logic signed [BUS_WIDTH-1:0] x,
                    output logic signed [BUS_WIDTH-1:0] ans
    );
    
    logic signed [BUS_WIDTH-1:0] x_sq, mul1_op1, mul1_op2, mul1_ans, mul2_op1, mul2_op2, mul2_ans;
    
    always_comb begin
        mul1_op1 = 32'hffffffff;
        mul1_op2 = 32'hffffffff;
        mul2_op1 = 32'hffffffff;
        mul2_op2 = 32'hffffffff;
        if (-64 < x && x < 64) begin 
            mul1_op1 = 15;
            mul1_op2 = x;
            ans = mul1_ans + 32;
        end else if (-128 <= x && x <= -64) begin 
            mul1_op1 = 3;
            mul1_op2 = x_sq;
            mul2_op1 = 19;
            mul2_op2 = x;
            ans = mul1_ans + mul2_ans + 33;
        end else if (-192 <= x && x < -128) begin 
            mul1_op1 = 2;
            mul1_op2 = x_sq;
            mul2_op1 = 14;
            mul2_op2 = x;
            ans = mul1_ans + mul2_ans + 28;
        end else if (-256 <= x && x < -192) begin 
            mul1_op1 = 1;
            mul1_op2 = x_sq;
            mul2_op1 = 8;
            mul2_op2 = x;
            ans = mul1_ans + mul2_ans + 19;
        end else if (-320 <= x && x < -256) begin 
//            mul1_op1 = 181193;
//            mul1_op2 = x_sq;
            mul2_op1 = 4;
            mul2_op2 = x;
            ans = mul2_ans + 11;
        end else if (x < -320) begin 
            ans = 0;
        end else if (64 <= x && x < 128) begin 
            mul1_op1 = -3;
            mul1_op2 = x_sq;
            mul2_op1 = 19;
            mul2_op2 = x;
            ans = mul1_ans + mul2_ans + 31;
        end else if (128 <= x && x < 192) begin 
            mul1_op1 = -2;
            mul1_op2 = x_sq;
            mul2_op1 = 14;
            mul2_op2 = x;
            ans = mul1_ans + mul2_ans + 36;
        end else if (192 <= x && x < 256) begin 
            mul1_op1 = -1;
            mul1_op2 = x_sq;
            mul2_op1 = 8;
            mul2_op2 = x;
            ans = mul1_ans + mul2_ans + 45;
        end else if (256 <= x && x < 320) begin 
//            mul1_op1 = -181193;
//            mul1_op2 = x_sq;
            mul2_op1 = 4;
            mul2_op2 = x;
            ans = mul2_ans + 53;
        end else if (320 <= x) begin 
            ans = 64;
        end else
            ans = 32'hffffffff;
    end
    
    fixed_point_mul #(
        .BUS_WIDTH(BUS_WIDTH),
        .NUM_DECIMAL_IN_BINARY(NUM_DECIMAL_IN_BINARY)
    ) mul1 (
        .a(mul1_op1),
        .b(mul1_op2),
        .ans(mul1_ans)
    );
    
    fixed_point_mul #(
        .BUS_WIDTH(BUS_WIDTH),
        .NUM_DECIMAL_IN_BINARY(NUM_DECIMAL_IN_BINARY)
    ) mul2 (
        .a(mul2_op1),
        .b(mul2_op2),
        .ans(mul2_ans)
    );
    
    fixed_point_mul #(
        .BUS_WIDTH(BUS_WIDTH),
        .NUM_DECIMAL_IN_BINARY(NUM_DECIMAL_IN_BINARY)
    ) mul3 (
        .a(x),
        .b(x),
        .ans(x_sq)
    );
endmodule