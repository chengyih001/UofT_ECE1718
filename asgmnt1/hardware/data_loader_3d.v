`timescale 1ns / 1ps

module data_loader_3d #(parameter BUS_WIDTH = 32,
                        parameter DEPTH = 8,
                        parameter HEIGHT = 8,
                        parameter WIDTH = 8) (
        input logic clk,
        input logic rst_n,
        input logic load,
        input logic signed [BUS_WIDTH-1:0] in_bus,
        output logic signed [BUS_WIDTH-1:0] signal_out[DEPTH][HEIGHT][WIDTH]
    );
    
    logic [9:0] counter_i, counter_j, counter_k;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            counter_i <= 10'b0;
            counter_j <= 10'b0;
            counter_k <= 10'b0;
            for (int i=0; i<DEPTH; i++)
                for (int j=0; j<HEIGHT; j++)
                    for (int k=0; k<WIDTH; k++)
                        signal_out[i][j][k] <= {(BUS_WIDTH-1){1'b0}};
        end else begin
            if (load) begin
                signal_out[counter_i][counter_j][counter_k] <= in_bus;
                // update the counters
                    if (counter_k >= WIDTH-1) begin
                        counter_k <= 10'b0;
                        if (counter_j >= HEIGHT-1) begin
                            counter_j <= 10'b0;
                            if (counter_i < DEPTH-1)
                                counter_i <= counter_i + 10'b1;
                        end else
                            counter_j <= counter_j + 10'b1;
                    end else
                        counter_k <= counter_k + 10'b1;
            end
        end
    end
    
endmodule
