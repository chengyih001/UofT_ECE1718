`timescale 1ns / 1ps

module carry_skip_adder_4bit (a, b, c_in, sum, c_out);

    input [3:0] a;
    input [3:0] b;
    input c_in;
    output [3:0] sum;
    output c_out;
    
    wire c_skip;
    wire [3:0] p;
    
    assign p = a ^ b;
    assign c_skip = &p;
    
    wire full_adder_c_out[4:0];
	
    genvar i;
    generate
        for (i=0; i<4; i=i+1) begin: adder_gen
            full_adder_1bit full_adder0 (
                .a(a[i]),
                .b(b[i]),
                .c_in(i == 0 ? c_in : full_adder_c_out[i]),
                .sum(sum[i]),
                .c_out(full_adder_c_out[i+1])
            );
        end
    endgenerate
    
	assign full_adder_c_out[0] = 1'b0; // not used
    assign c_out = c_skip ? c_in : full_adder_c_out[4];

endmodule
