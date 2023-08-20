`timescale 1ns / 1ps

module fixed_point_mul #(parameter BUS_WIDTH = 32,
                         parameter NUM_DECIMAL_IN_BINARY = 6)(
        input logic signed [BUS_WIDTH-1:0] a,
        input logic signed [BUS_WIDTH-1:0] b,
        output logic signed [BUS_WIDTH-1:0] ans
    );
    
    logic signed [BUS_WIDTH+BUS_WIDTH -1:0] test, pos_test;
    logic signed [BUS_WIDTH:0] shifted_pos_test;
    logic signed [BUS_WIDTH-1:0] temp;
    logic lsb;
    
    assign test = a * b;
    assign pos_test = test[BUS_WIDTH+BUS_WIDTH -1] == 1'b1 ? (64'hffffffffffffffff ^ test) + 64'b1 : test;
    assign shifted_pos_test = pos_test >>> (NUM_DECIMAL_IN_BINARY-1);
    
    assign lsb = shifted_pos_test[0];
    assign temp = ((shifted_pos_test >>> 1) + lsb);
    
    assign ans = test[BUS_WIDTH+BUS_WIDTH -1] == 1'b1 ? ((32'hffffffff ^ temp) + 32'b1) : temp;
    
endmodule
