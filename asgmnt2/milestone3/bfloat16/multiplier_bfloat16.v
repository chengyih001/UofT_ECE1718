`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 03/15/2023 02:04:50 PM
// Design Name: 
// Module Name: multiplier
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module multiplier_bfloat16 (
    input [15:0] n1,
    input [15:0] n2,
    output [15:0] res,
    output Overflow,
    output Underflow,
    output Exception
);

wire and_E1, and_E2, or_E1, or_E2, Sign_final, E_carry, round_0, round_1, round_final, temp;
wire [15:0] M_res;
wire [6:0] M_final;
wire [7:0] M_normalized;
wire [8:0] E_res, E_final;

// Check edge cases

// 1. All exponent bits are set to 1 --> inf or NANA
reductionAND_8bits AND0(.in(n1[14:7]), .out(and_E1));
reductionAND_8bits AND1(.in(n2[14:7]), .out(and_E2));
assign Exception = and_E1 | and_E2;
// 2. All exponent bits are set to 0 --> mantissa == 0
reductionOR_8bits OR0(.in(n1[14:7]), .out(or_E1));
reductionOR_8bits OR1(.in(n2[14:7]), .out(or_E2));

// Final sign
assign Sign_final = n1[15] ^ n2[15];

// Multiply mantissas
Multiply_8bits MUL_8_0(.a({or_E1, n1[6:0]}), .b({or_E2, n2[6:0]}), .out(M_res));

// Rounding mantissa
reductionOR_8bits OR2(.in({1'b0, M_res[6:0]}), .out(round_0));
reductionOR_8bits OR3(.in(M_res[7:0]), .out(round_1));
Mux_1bit Mux_1_0(.a(round_0), .b(round_1), .compare(M_res[15]), .out(round_final));

// Normalize mantissa
Mux_8bits Mux_8_0(.a({1'b0, M_res[13:7]}), .b({1'b0, M_res[14:8]}), .compare(M_res[15]), .out(M_normalized));
Adder_8bits ADD_8_0(.a({1'b0, M_normalized[6:0]}), .b({7'b0, round_final}), .cin(1'b0), .out({temp, M_final}), .cout());

// Add Exponents then subtract 127 to compensate for bias
Adder_8bits Add_8_0(.a(n1[14:7]), .b(n2[14:7]), .cin(1'b0), .out(E_res[7:0]), .cout(E_res[8]));
Adder_9bits Add_9_0(.a(E_res), .b(9'b110000001), .cin(M_res[15]), .out(E_final), .cout(E_carry));

// Concatenate final answer (Sign + Exponent + Mantissa)
assign res = {Sign_final, E_final[7:0], M_final};
assign Underflow = !E_carry;
assign Overflow = E_carry & E_final[8];

endmodule