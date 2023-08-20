`timescale 1ns / 1ps

module dense_layer_2 #(parameter BUS_WIDTH = 32,
                      parameter NUM_DECIMAL_IN_BINARY = 6)  (
        input logic clk,
        input logic rst_n,
        input logic start,
        input logic signed [9:0] in_bus[120],
        input logic signed [9:0] dense_w[120][10],
        input logic signed [BUS_WIDTH-1:0] dense_b[10],
        output logic done,
        output logic signed [BUS_WIDTH-1:0] out_bus[10]
    );
    
    logic signed [BUS_WIDTH-1:0] temp[10], dense_layer[10];
    
    enum {IDLE, COMPUTING, DONE} state;
    logic [7:0] counter_j;
    logic signed [BUS_WIDTH-1:0] mul_ans[9:0];
    logic signed [BUS_WIDTH-1:0] pp_0[4:0];
    logic signed [BUS_WIDTH-1:0] pp_1[2:0];
    logic signed [BUS_WIDTH-1:0] pp_2[1:0];
    logic signed [BUS_WIDTH-1:0] pp_3;
    
    
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            done <= 1'b0;
            counter_j <= 8'b0;
            for (int i=0; i < 10; i++) begin
                temp[i] <= {(BUS_WIDTH-1){1'b0}};
                dense_layer[i] <= {(BUS_WIDTH-1){1'b0}};
                out_bus[i] <= {(BUS_WIDTH-1){1'b0}};
            end
        end
        
        else begin
            case(state)
                IDLE: begin
                    `ifdef NO_IDLE_RESET
                    done <= 1'b0;
                    counter_j <= 8'b0;
                    
                    for (int i=0; i < 10; i++) begin
                        temp[i] <= {(BUS_WIDTH-1){1'b0}};
                        dense_layer[i] <= {(BUS_WIDTH-1){1'b0}};
                        out_bus[i] <= {(BUS_WIDTH-1){1'b0}};
                    end
                    `endif
                    if (start == 1'b1) begin
                        state <= COMPUTING;
                    end
                end
                
                COMPUTING: begin
//                    for (int j=0; j<120; j++) {
//                        for (int i=0; i<10; i++) {
//                            dense_sum2[i] += dense_w2[j][i] * dense2_input[j];
//                        }
//                    }
//                    for (int i=0; i < 10; i++) {
//                        dense_sum2[i] += dense_b2[i];
//                    }
                    
                    for (int i=0; i < 10; i++) begin
                        dense_layer[i] <= dense_layer[i] + mul_ans[i];
                    end
                    
                    if (counter_j >= 119) begin
                        state <= DONE;
                    end
                    else begin
                        counter_j <= counter_j + 1;
                    end
                    
                end
                
                DONE: begin
                    for (int i=0; i < 10; i++) begin
                        temp[i] <= dense_layer[i] + dense_b[i];
                        out_bus[i] <= dense_layer[i];
                    end
                    done <= 1'b1;
                    state <= IDLE;
                end
                
                default: state <= IDLE;
                
            endcase
        end
    end
    
    genvar c_i;
    generate
        for (c_i=0; c_i<10; c_i++) begin
            fixed_point_mul #(
                .BUS_WIDTH(BUS_WIDTH),
                .NUM_DECIMAL_IN_BINARY(NUM_DECIMAL_IN_BINARY)
            ) mul0 (
                .a({{(BUS_WIDTH - 10){in_bus[counter_j][9]}}, in_bus[counter_j]}),
                .b({{(BUS_WIDTH - 10){dense_w[counter_j][c_i][9]}}, dense_w[counter_j][c_i]}),
                .ans(mul_ans[c_i])
            );
        end
    endgenerate
    
endmodule