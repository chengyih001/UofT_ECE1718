`timescale 1ns / 1ps

module data_loader_1d #(parameter BUS_WIDTH = 32,
                        parameter WIDTH = 8) (
        input logic clk,
        input logic rst_n,
        input logic load,
        input logic signed [BUS_WIDTH-1:0] in_bus,
        output logic signed [BUS_WIDTH-1:0] signal_out[WIDTH]
    );
    
    logic [9:0] counter_i;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            counter_i <= 10'b0;
            for (int i=0; i<WIDTH; i++)
                signal_out[i] <= {(BUS_WIDTH-1){1'b0}};
        end else begin
            if (load) begin
                signal_out[counter_i] <= in_bus;
                // update the counters
                if (counter_i < WIDTH-1)
                    counter_i <= counter_i + 10'b1;
            end
        end
    end
    
endmodule
