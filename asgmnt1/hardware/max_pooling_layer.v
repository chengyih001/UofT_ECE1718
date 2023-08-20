`timescale 1ns / 1ps

module max_pooling_layer #(parameter BUS_WIDTH = 32,
                           parameter NUM_DECIMAL_IN_BINARY = 6) (
        input logic clk,
        input logic rst_n,
        input logic start,
        input logic signed [9:0] sig_layer[5][28][28],
        output logic signed [9:0] max_layer[5][14][14], // dense_input
        output logic done
    );
    
    logic signed [BUS_WIDTH-1:0] cur_max;
    logic [4:0] filter_dim, counter_i, counter_j;
    logic [3:0] counter_i_d2, counter_j_d2;
    
    enum {IDLE, COMPUTING, DONE} state;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            filter_dim <= 5'b0;
            counter_i <= 5'b0;
            counter_j <= 5'b0;
            done <= 1'b0;
            for (int i=0; i<5; i++) begin
                for (int j=0; j<14; j++) begin
                    for (int k=0; k<14; k++) begin
                        max_layer[i][j][k] <= {(BUS_WIDTH-1){1'b0}};
                    end
                end
            end
        end else begin
            case (state)
                IDLE: begin
                    `ifdef NO_IDLE_RESET
                    filter_dim <= 5'b0;
                    counter_i <= 5'b0;
                    counter_j <= 5'b0;
                    done <= 1'b0;
                    for (int i=0; i<5; i++) begin
                        for (int j=0; j<14; j++) begin
                            for (int k=0; k<14; k++) begin
                                dense_input[i][j][k] <= {(BUS_WIDTH-1){1'b0}};
                            end
                        end
                    end
                    `endif
                    if (start == 1'b1) begin
                        state <= COMPUTING;
                    end
                end
                COMPUTING: begin
                    max_layer[filter_dim][counter_i_d2][counter_j_d2] <= cur_max;
                    if (counter_j >= 26) begin
                        counter_j <= 5'b0;
                        if (counter_i >= 26) begin
                            counter_i <= 5'b0;
                            if (filter_dim >= 4) begin
                                // done computing
                                state <= DONE;
                            end else 
                                filter_dim <= filter_dim + 1;
                        end else 
                            counter_i <= counter_i + 2;
                    end else
                        counter_j <= counter_j + 2;
                end
                DONE: begin
                    done <= 1'b1;
                    if (start == 1'b0) // use dropping of start signal to indecate of done and reset the current state
                        state <= IDLE;
                end
                default: state <= IDLE;
            endcase
        end
    end
    
    logic signed [BUS_WIDTH-1:0] temp_max1, temp_max2;
    always_comb begin
        temp_max1 = (sig_layer[filter_dim][counter_i][counter_j] > sig_layer[filter_dim][counter_i][counter_j+1]) ? 
                     {{(BUS_WIDTH - 10){sig_layer[filter_dim][counter_i][counter_j][9]}}, sig_layer[filter_dim][counter_i][counter_j]} : 
                     {{(BUS_WIDTH - 10){sig_layer[filter_dim][counter_i][counter_j+1][9]}}, sig_layer[filter_dim][counter_i][counter_j+1]};
        temp_max2 = (sig_layer[filter_dim][counter_i+1][counter_j] > sig_layer[filter_dim][counter_i+1][counter_j+1]) ? 
                     {{(BUS_WIDTH - 10){sig_layer[filter_dim][counter_i+1][counter_j][9]}}, sig_layer[filter_dim][counter_i+1][counter_j]} : 
                     {{(BUS_WIDTH - 10){sig_layer[filter_dim][counter_i+1][counter_j+1][9]}}, sig_layer[filter_dim][counter_i+1][counter_j+1]};
        cur_max = (temp_max1 > temp_max2) ? temp_max1 : temp_max2;
        
        counter_i_d2 = counter_i >>> 1;
        counter_j_d2 = counter_j >>> 1;
    end
    
endmodule