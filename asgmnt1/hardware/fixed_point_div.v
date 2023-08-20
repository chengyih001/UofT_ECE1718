`timescale 1ns / 1ps

module fixed_point_div #(parameter BUS_WIDTH = 32,
                         parameter NUM_DECIMAL_IN_BINARY = 6)(
                         input logic signed [BUS_WIDTH-1:0] a,
                         input logic signed [BUS_WIDTH-1:0] b,
                         output logic signed [BUS_WIDTH-1:0] ans
    );
    
    logic signed [BUS_WIDTH+NUM_DECIMAL_IN_BINARY-1:0] temp_a, temp_b;
    logic signed [BUS_WIDTH-1:0] temp_ans, pos_temp_ans, pos_ans;
    logic lsb;
    
    assign temp_a = a << (NUM_DECIMAL_IN_BINARY+1);
    assign temp_b = {{NUM_DECIMAL_IN_BINARY{1'b0}}, b};
    
    assign temp_ans = temp_a / temp_b;
    assign pos_temp_ans = temp_ans[BUS_WIDTH-1] == 1'b1 ? (32'hffffffff ^ temp_ans + 32'b1) : (temp_ans);
    assign lsb = temp_ans[0];
    assign pos_ans = (pos_temp_ans >> 1) + lsb;
    assign ans = temp_ans[BUS_WIDTH-1] == 1'b1 ? (32'hffffffff ^ pos_ans + 32'b1) : (pos_ans);
endmodule
