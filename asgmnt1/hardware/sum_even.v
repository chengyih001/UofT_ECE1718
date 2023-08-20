`timescale 1ns / 1ps

module sum_even #(parameter BUS_WIDTH = 32,
                  parameter NUM_ELEMENTS = 10 )(
        input logic signed [BUS_WIDTH-1:0] input_signal[NUM_ELEMENTS],
        output logic signed [BUS_WIDTH-1:0] sum
    );
    
    logic signed [BUS_WIDTH-1:0] temp_sum[NUM_ELEMENTS];
    
    always_comb begin
        for (int i=0; i<NUM_ELEMENTS; i++) begin
            if (i == 0) begin
                temp_sum[i] = input_signal[i];
            end else begin
                temp_sum[i] = temp_sum[i-1] + input_signal[i];
            end
        end
    end
    
    assign sum = temp_sum[NUM_ELEMENTS-1];
endmodule
