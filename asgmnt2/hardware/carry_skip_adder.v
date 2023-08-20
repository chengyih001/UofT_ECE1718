`timescale 1ns / 1ps

module carry_skip_adder (a, b, sum, c_out);

    input signed [31:0] a;
    input signed [31:0] b;
    output signed [31:0] sum;
    output c_out;
    
    wire [8:0] temp_c;
    
    genvar i;
    generate
        for (i=0; i<8; i=i+1) begin: adder_gen
            carry_skip_adder_4bit adder_4bit (
                .a(a[(i+1)*4-1 : i*4]),
                .b(b[(i+1)*4-1 : i*4]),
                .c_in(i == 0 ? 1'b0 : temp_c[i]),
                .sum(sum[(i+1)*4-1 : i*4]),
                .c_out(temp_c[i+1])
            );
        end
    endgenerate

	assign temp_c[0] = 1'b0; // not used
    assign c_out = temp_c[8];

endmodule
