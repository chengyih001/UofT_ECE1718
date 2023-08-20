`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 03/13/2023 10:47:37 AM
// Design Name: 
// Module Name: utils
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


module reductionAND_8bits (input [7:0] in, output out);
    wire w1, w2, w3, w4, w5, w6;
    assign w1 = in[1] & in[0];
    assign w2 = in[2] & w1;
    assign w3 = in[3] & w2;
    assign w4 = in[4] & w3;
    assign w5 = in[5] & w4;
    assign w6 = in[6] & w5;
    assign out = in[7] & w6;
endmodule

module reductionOR_8bits (input [7:0] in, output out);
    wire w1, w2, w3, w4, w5, w6;
    assign w1 = in[1] | in[0];
    assign w2 = in[2] | w1;
    assign w3 = in[3] | w2;
    assign w4 = in[4] | w3;
    assign w5 = in[5] | w4;
    assign w6 = in[6] | w5;
    assign out = in[7] | w6;
endmodule

module reductionOR_24bits (input [23:0] in, output out);
    wire w1, w2, w3;
    reductionOR_8bits OR_0(.in(in[7:0]), .out(w1));
    reductionOR_8bits OR_1(.in(in[15:8]), .out(w2));
    reductionOR_8bits OR_2(.in(in[23:16]), .out(w3));
    assign out = w1 | w2 | w3;
endmodule

module Complement_1s_8bits (input [7:0] in, output [7:0] out);
    assign out[0] = !in[0];
    assign out[1] = !in[1];
    assign out[2] = !in[2];
    assign out[3] = !in[3];
    assign out[4] = !in[4];
    assign out[5] = !in[5];
    assign out[6] = !in[6];
    assign out[7] = !in[7];
endmodule

module Complement_1s_24bits (input [23:0] in, output [23:0] out);
    Complement_1s_8bits COMP0(.in(in[7:0]), .out(out[7:0]));
    Complement_1s_8bits COMP1(.in(in[15:8]), .out(out[15:8]));
    Complement_1s_8bits COMP2(.in(in[23:16]), .out(out[23:16]));
endmodule

module Complement_2s_8bits (input [7:0] in, output [7:0] out);
    wire [7:0] temp;
    wire dummy;
    Complement_1s_8bits COMP0(.in(in), .out(temp));
    Adder_8bits ADD0(.a(temp), .b(8'b0000_0001), .cin(1'b0), .out(out), .cout(dummy));
endmodule

module Complement_2s_24bits (input [23:0] in, output [23:0] out);
    wire [23:0] temp;
    wire dummy;
    Complement_1s_24bits COMP0(.in(in), .out(temp));
    Adder_24bits ADD0(.a(temp), .b(24'b0000_0000_0000_0000_0000_0001), .cin(1'b0), .out(out), .cout(dummy));
endmodule

module Adder_4bits(input [3:0] a, input [3:0] b, input cin, output [3:0] out, output cout);
    wire p0, p1, p2, p3;
    wire g0, g1, g2, g3;
    wire c0, c1, c2;
       
    assign p0 = a[0] ^ b[0];
    assign p1 = a[1] ^ b[1];
    assign p2 = a[2] ^ b[2];
    assign p3 = a[3] ^ b[3];
    assign g0 = a[0] & b[0];
    assign g1 = a[1] & b[1];
    assign g2 = a[2] & b[2];
    assign g3 = a[3] & b[3];
    
    assign c0 = g0 | (p0 & cin);
    assign c1 = g1 | (p1 & g0) | (p1 & p0 & cin);
    assign c2 = g2 | (p2 & g1) | (p2 & p1 & g0) | (p2 & p1 & p0 & cin);
    assign cout = g3 | (p3 & g2) | (p3 & p2 & g1) | (p3 & p2 & p1 & g0) | (p3 & p2 & p1 & p0 & cin);
    
    assign out[0] = p0 ^ cin;
    assign out[1] = p1 ^ c0;
    assign out[2] = p2 ^ c1;
    assign out[3] = p3 ^ c2;
endmodule

module Adder_8bits(input [7:0] a, input [7:0] b, input cin, output [7:0] out, output cout);
    wire cout_0;
    Adder_4bits ADD0(.a(a[3:0]), .b(b[3:0]), .cin(cin), .out(out[3:0]), .cout(cout_0));
    Adder_4bits ADD1(.a(a[7:4]), .b(b[7:4]), .cin(cout_0), .out(out[7:4]), .cout(cout));
endmodule

module Adder_9bits(input [8:0] a, input [8:0] b, input cin, output [8:0] out, output cout);
    wire cout_0;
    Adder_8bits ADD0(.a(a[7:0]), .b(b[7:0]), .cin(cin), .out(out[7:0]), .cout(cout_0));
    assign out[8] = a[8] ^ b[8] ^ cout_0;
    assign cout = a[8] & b[8] | a[8] & cout_0 | cout_0 & b[8];
endmodule

module Adder_24bits(input [23:0] a, input [23:0] b, input cin, output [23:0] out, output cout);
    wire cout_0, cout_1;
    Adder_8bits ADD0(.a(a[7:0]), .b(b[7:0]), .cin(cin), .out(out[7:0]), .cout(cout_0));
    Adder_8bits ADD1(.a(a[15:8]), .b(b[15:8]), .cin(cout_0), .out(out[15:8]), .cout(cout_1));
    Adder_8bits ADD2(.a(a[23:16]), .b(b[23:16]), .cin(cout_1), .out(out[23:16]), .cout(cout));
endmodule

module Multiply_8bits(input [7:0] a, input [7:0] b, output [15:0] out);
    assign out = a*b;
endmodule

module Multiply_24bits(input [23:0] a, input [23:0] b, output [47:0] out);
    assign out = a*b;
endmodule

module Mux_1bit(input a, input b, input compare, output out);
    wire w1,w2;
	assign w1 = a & !compare;
	assign w2 = b & compare;
	assign out = w1 | w2;
endmodule

module Mux_8bits(input [7:0] a, input [7:0] b, input compare, output [7:0] out);
    Mux_1bit M0(.a(a[0]), .b(b[0]), .compare(compare), .out(out[0]));
    Mux_1bit M1(.a(a[1]), .b(b[1]), .compare(compare), .out(out[1]));
    Mux_1bit M2(.a(a[2]), .b(b[2]), .compare(compare), .out(out[2]));
    Mux_1bit M3(.a(a[3]), .b(b[3]), .compare(compare), .out(out[3]));
    Mux_1bit M4(.a(a[4]), .b(b[4]), .compare(compare), .out(out[4]));
    Mux_1bit M5(.a(a[5]), .b(b[5]), .compare(compare), .out(out[5]));
    Mux_1bit M6(.a(a[6]), .b(b[6]), .compare(compare), .out(out[6]));
    Mux_1bit M7(.a(a[7]), .b(b[7]), .compare(compare), .out(out[7]));
endmodule

module Mux_24bits(input [23:0] a, input [23:0] b, input compare, output [23:0] out);
    Mux_8bits M0(.a(a[7:0]), .b(b[7:0]), .compare(compare), .out(out[7:0]));
    Mux_8bits M1(.a(a[15:8]), .b(b[15:8]), .compare(compare), .out(out[15:8]));
    Mux_8bits M2(.a(a[23:16]), .b(b[23:16]), .compare(compare), .out(out[23:16]));
endmodule

module Normalize_and_Shift(input [23:0] in, input carry, input oper, output reg [4:0] shift, output reg [22:0] out);
    reg [23:0] temp;
    
    always @(*)
    begin
        if (carry & !oper)
        begin
            out = in[23:1] + {22'b0, in[0]};
            shift = 5'd0;
        end
        else
        begin
            casex(in)
                24'b1xxx_xxxx_xxxx_xxxx_xxxx_xxxx:
                begin
                    out = in[22:0];
                    shift = 5'd0;
                end
                24'b01xx_xxxx_xxxx_xxxx_xxxx_xxxx:
                begin
                    temp = in << 1;
                    out = temp[22:0];
                    shift = 5'd1;
                end
                24'b001x_xxxx_xxxx_xxxx_xxxx_xxxx:
                begin
                    temp = in << 2;
                    out = temp[22:0];
                    shift = 5'd2;
                end
                24'b0001_xxxx_xxxx_xxxx_xxxx_xxxx:
                begin
                    temp = in << 3;
                    out = temp[22:0];
                    shift = 5'd3;
                end
                24'b0000_1xxx_xxxx_xxxx_xxxx_xxxx:
                begin
                    temp = in << 4;
                    out = temp[22:0];
                    shift = 5'd4;
                end
                24'b0000_01xx_xxxx_xxxx_xxxx_xxxx:
                begin
                    temp = in << 5;
                    out = temp[22:0];
                    shift = 5'd5;
                end
                24'b0000_001x_xxxx_xxxx_xxxx_xxxx:
                begin
                    temp = in << 6;
                    out = temp[22:0];
                    shift = 5'd6;
                end
                24'b0000_0001_xxxx_xxxx_xxxx_xxxx:
                begin
                    temp = in << 7;
                    out = temp[22:0];
                    shift = 5'd7;
                end
                24'b0000_0000_1xxx_xxxx_xxxx_xxxx:
                begin
                    temp = in << 8;
                    out = temp[22:0];
                    shift = 5'd8;
                end
                24'b0000_0000_01xx_xxxx_xxxx_xxxx:
                begin
                    temp = in << 9;
                    out = temp[22:0];
                    shift = 5'd9;
                end
                24'b0000_0000_001x_xxxx_xxxx_xxxx:
                begin
                    temp = in << 10;
                    out = temp[22:0];
                    shift = 5'd10;
                end
                24'b0000_0000_0001_xxxx_xxxx_xxxx:
                begin
                    temp = in << 11;
                    out = temp[22:0];
                    shift = 5'd11;
                end
                24'b0000_0000_0000_1xxx_xxxx_xxxx:
                begin
                    temp = in << 12;
                    out = temp[22:0];
                    shift = 5'd12;
                end
                24'b0000_0000_0000_01xx_xxxx_xxxx:
                begin
                    temp = in << 13;
                    out = temp[22:0];
                    shift = 5'd13;
                end
                24'b0000_0000_0000_001x_xxxx_xxxx:
                begin
                    temp = in << 14;
                    out = temp[22:0];
                    shift = 5'd14;
                end
                24'b0000_0000_0000_0001_xxxx_xxxx:
                begin
                    temp = in << 15;
                    out = temp[22:0];
                    shift = 5'd15;
                end
                24'b0000_0000_0000_0000_1xxx_xxxx:
                begin
                    temp = in << 16;
                    out = temp[22:0];
                    shift = 5'd16;
                end
                24'b0000_0000_0000_0000_01xx_xxxx:
                begin
                    temp = in << 17;
                    out = temp[22:0];
                    shift = 5'd17;
                end
                24'b0000_0000_0000_0000_001x_xxxx:
                begin
                    temp = in << 18;
                    out = temp[22:0];
                    shift = 5'd18;
                end
                24'b0000_0000_0000_0000_0001_xxxx:
                begin
                    temp = in << 19;
                    out = temp[22:0];
                    shift = 5'd19;
                end
                24'b0000_0000_0000_0000_0000_1xxx:
                begin
                    temp = in << 20;
                    out = temp[22:0];
                    shift = 5'd20;
                end
                24'b0000_0000_0000_0000_0000_01xx:
                begin
                    temp = in << 21;
                    out = temp[22:0];
                    shift = 5'd21;
                end
                24'b0000_0000_0000_0000_0000_001x:
                begin
                    temp = in << 22;
                    out = temp[22:0];
                    shift = 5'd22;
                end
                24'b0000_0000_0000_0000_0000_0001:
                begin
                    temp = in << 23;
                    out = temp[22:0];
                    shift = 5'd23;
                end
                default:
                begin
                    out = 23'b0;
                    shift = 5'd0;
                end     
            endcase
        end
    end
endmodule

module Normalize_and_Shift_bfloat16(input [7:0] in, input carry, input oper, output reg [4:0] shift, output reg [6:0] out);
    reg [7:0] temp;
    
    always @(*)
    begin
        if (carry & !oper)
        begin
            out = in[7:1] + {6'b0, in[0]};
            shift = 5'd0;
        end
        else
        begin
            casex(in)
                8'b1xxx_xxxx:
                begin
                    out = in[6:0];
                    shift = 5'd0;
                end
                8'b01xx_xxxx:
                begin
                    temp = in << 1;
                    out = temp[6:0];
                    shift = 5'd1;
                end
                8'b001x_xxxx:
                begin
                    temp = in << 2;
                    out = temp[6:0];
                    shift = 5'd2;
                end
                8'b0001_xxxx:
                begin
                    temp = in << 3;
                    out = temp[6:0];
                    shift = 5'd3;
                end
                8'b0000_1xxx:
                begin
                    temp = in << 4;
                    out = temp[6:0];
                    shift = 5'd4;
                end
                8'b0000_01xx:
                begin
                    temp = in << 5;
                    out = temp[6:0];
                    shift = 5'd5;
                end
                8'b0000_001x:
                begin
                    temp = in << 6;
                    out = temp[6:0];
                    shift = 5'd6;
                end
                8'b0000_0001:
                begin
                    temp = in << 7;
                    out = temp[6:0];
                    shift = 5'd7;
                end
                default:
                begin
                    out = 7'b0;
                    shift = 5'd0;
                end     
            endcase
        end
    end
endmodule