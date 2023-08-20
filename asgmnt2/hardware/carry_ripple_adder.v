`timescale 1ns / 1ps

module carry_ripple_adder (a, b, sum, c_out);
    input signed [32-1:0] a;
    input signed [32-1:0] b;
    output signed [32-1:0] sum;
    output c_out;

    wire half_adder_c_out;
    wire full_adder_c_out[32-2:0];
    
    half_adder_1bit half_adder (
        .a(a[0]),
        .b(b[0]),
        .sum(sum[0]),
        .c_out(half_adder_c_out)
    );
    
    full_adder_1bit full_adder0 (
        .a(a[1]),
        .b(b[1]),
        .c_in(half_adder_c_out),
        .sum(sum[1]),
        .c_out(full_adder_c_out[0])
    );
    
    generate
		  genvar i;
        for (i=1; i<32-1; i=i+1) begin: full_adder_gen
            full_adder_1bit full_adder1 (
                .a(a[i+1]),
                .b(b[i+1]),
                .c_in(full_adder_c_out[i-1]),
                .sum(sum[i+1]),
                .c_out(full_adder_c_out[i])
            );
        end
    endgenerate
    
    assign c_out = full_adder_c_out[32-2];
    
endmodule
