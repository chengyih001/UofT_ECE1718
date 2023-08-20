`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 03/13/2023 10:40:45 AM
// Design Name: 
// Module Name: adder
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


module adder_bfloat16 (
    input [15:0] n1,
    input [15:0] n2,
    output [15:0] res,
    output Overflow,
    output Underflow,
    output Exception
);

wire and_E1, and_E2, or_E1, or_E2, E1_greater_than_E2, real_oper, real_sign, M_carry, w1, w2, w3, Sign_final;
wire [7:0] comp_E2, E1_E2_diff, t_E1_E2_diff, comp_t_E1_E2_diff, E_larger, One_Added_E, new_E, comp_E_shift;
wire [8:0] E_final;
wire [7:0] M1, M2, comp_M2, selected_M2, M_sum, comp_M_sum, M_result;
wire [6:0] M_final;
wire [4:0] E_shift;

// Check edge cases

// 1. All exponent bits are set to 1 --> inf or NANA
reductionAND_8bits AND0(.in(n1[14:7]), .out(and_E1));
reductionAND_8bits AND1(.in(n2[14:7]), .out(and_E2));
assign Exception = and_E1 | and_E2;
// 2. All exponent bits are set to 0 --> mantissa == 0
reductionOR_8bits OR0(.in(n1[14:7]), .out(or_E1));
reductionOR_8bits OR1(.in(n2[14:7]), .out(or_E2));

// Shift smaller number so that exponent matches larger number

// 1. E1 - E2 (E1 + E2's 2 complement)
Complement_1s_8bits COMP_1_0(.in(n2[14:7]), .out(comp_E2));
Adder_8bits ADD0(.a(n1[14:7]), .b(comp_E2), .cin(1'b1), .out(t_E1_E2_diff), .cout(E1_greater_than_E2));
// 2. Select original E1_E2_diff if E1 > E2; Select 2s complement if E2 > E1;
Complement_2s_8bits COMP_2_0(.in(t_E1_E2_diff), .out(comp_t_E1_E2_diff));
Mux_8bits MUX0(.a(comp_t_E1_E2_diff), .b(t_E1_E2_diff), .compare(E1_greater_than_E2), .out(E1_E2_diff));
// 3. Select larger exponent
Mux_8bits MUX1(.a(n2[14:7]), .b(n1[14:7]), .compare(E1_greater_than_E2), .out(E_larger));
// 4. Shift mantissa
assign M1 = E1_greater_than_E2? {or_E1, n1[6:0]} : {or_E1, n1[6:0]} >> E1_E2_diff;
assign M2 = E1_greater_than_E2? {or_E2, n2[6:0]} >> E1_E2_diff : {or_E2, n2[6:0]};

// Add mantissas
assign real_oper = n1[15] ^ n2[15];
assign real_sign = n1[15];

Complement_1s_8bits COMP_1_1(.in(M2), .out(comp_M2));
Mux_8bits MUX2(.a(M2), .b(comp_M2), .compare(real_oper), .out(selected_M2));
Adder_8bits ADD1(.a(M1), .b(selected_M2), .cin(real_oper), .out(M_sum), .cout(M_carry));

assign w1 = ~real_sign & real_oper & ~M_carry;
assign w2 = ~real_oper & real_sign;
assign w3 = M_carry & real_sign;
assign Sign_final = w1 | w2 | w3;

Adder_8bits ADD2(.a(E_larger), .b(8'b1), .cin(1'b0), .out(One_Added_E), .cout());
Mux_8bits MUX3(.a(E_larger), .b(One_Added_E), .compare(M_carry & !real_oper), .out(new_E));

// if M_sum < 0, select 2's complement of M_sum
Complement_2s_8bits COMP_2_1(.in(M_sum), .out(comp_M_sum));
Mux_8bits MUX4(.a(M_sum), .b(comp_M_sum), .compare(real_oper&!M_carry), .out(M_result));

// Normalize mantissa
Normalize_and_Shift_bfloat16 NS_0(.in(M_result), .carry(M_carry), .oper(real_oper), .out(M_final), .shift(E_shift));

// Shift Exponent
Complement_1s_8bits COMP1_2(.in({3'b000, E_shift}), .out(comp_E_shift));
Adder_8bits ADD3(.a(new_E), .b(comp_E_shift), .cin(1'b1), .out(E_final[7:0]), .cout(E_final[8]));

// Concatenate final answer (Sign + Exponent + Mantissa)
assign res = {Sign_final, E_final[7:0], M_final};
assign Underflow = !E_final[8];
assign Overflow = (&One_Added_E) && (~|E_shift);

endmodule