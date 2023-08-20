`timescale 1ns / 1ps

module convolution_layer #(parameter BUS_WIDTH = 32,
                           parameter NUM_DECIMAL_IN_BINARY = 6) (
        input logic clk,
        input logic rst_n,
        input logic start,
        input logic signed [14:0] img[1120],
        input logic signed [14:0] conv_w[5][7][7],
        input logic signed [9:0] conv_b[5][28][28],
        output logic signed [9:0] sig_layer[5][28][28],
        output logic done
    );
    
    enum {IDLE, COMPUTING, DONE} state;
    logic [4:0] filter_dim, counter_i, counter_j;
    logic [4:0] filter_dim_d, counter_i_d, counter_j_d, filter_dim_dd, counter_i_dd;
    logic signed [25:0] mul_ans[49];
    logic signed [9:0] sigmoid_out;
    logic signed [25:0] temp_sum_layer1[24], temp_sum_layer2[12], temp_sum_layer3[6], temp_sum_layer4[3], temp_sum_total;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            filter_dim_d <= filter_dim;
            counter_i_d <= counter_i;
            counter_j_d <= counter_j;
            done <= 1'b0;
            filter_dim <= 5'b0;
            counter_i <= 5'b0;
            counter_j <= 5'b0;
            for (int i=0; i<5; i++) begin
                for (int j=0; j<28; j++) begin
                    for (int k=0; k<28; k++) begin
                        sig_layer[i][j][k] <= {(BUS_WIDTH-1){1'b0}};
                    end
                end
            end
        end else begin
            filter_dim_d <= filter_dim;
            counter_i_d <= counter_i;
            counter_j_d <= counter_j;
            case (state)
                IDLE: begin
                    `ifdef NO_IDLE_RESET
                    done <= 1'b0;
                    filter_dim <= 5'b0;
                    counter_i <= 5'b0;
                    counter_j <= 5'b0;
                    counter_k <= 5'b0;
                    counter_l <= 5'b0;
                    for (int i=0; i<5; i++) begin
                        for (int j=0; j<28; j++) begin
                            for (int k=0; k<28; k++) begin
                                sig_layer[i][j][k] <= {(BUS_WIDTH-1){1'b0}};
                            end
                        end
                    end
                    `endif
                    if (start == 1'b1) begin
                        state <= COMPUTING;
                    end
                end
                COMPUTING: begin
                    sig_layer[filter_dim][counter_i][counter_j] <= sigmoid_out;
                    if (counter_j >= 27) begin
                        counter_j <= 0;
                        if (counter_i >= 27) begin
                            counter_i <= 0;
                            if (filter_dim >= 4) begin
                                state <= DONE;
                            end else begin
                                filter_dim <= filter_dim + 1;
                            end
                        end else begin
                            counter_i <= counter_i + 1;
                        end
                    end else begin
                        counter_j <= counter_j + 1;
                    end
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
    
    genvar k, l;
    generate
        for (k=0; k<7; k++) begin
            for (l=0; l<7; l++) begin
                fixed_point_mul #(
                    .BUS_WIDTH(26),
                    .NUM_DECIMAL_IN_BINARY(NUM_DECIMAL_IN_BINARY)
                ) mul0 (
                    .a({{(26 - 15){1'b0}}, img[(counter_i+k+1) * 32 + counter_j+l-2]}),
                    .b({{(26 - 15){conv_w[filter_dim][k][l][14]}}, conv_w[filter_dim][k][l]}),
                    .ans(mul_ans[k*7 + l])
                );
            end
        end
    endgenerate
    
    always_comb begin
        for (int i=0; i<24; i++) begin
            temp_sum_layer1[i] = mul_ans[i*2] + mul_ans[i*2+1];
        end
        for (int i=0; i<12; i++) begin
            temp_sum_layer2[i] = temp_sum_layer1[i*2] + temp_sum_layer1[i*2+1];
        end
        for (int i=0; i<6; i++) begin
            temp_sum_layer3[i] = temp_sum_layer2[i*2] + temp_sum_layer2[i*2+1];
        end
        for (int i=0; i<3; i++) begin
            temp_sum_layer4[i] = temp_sum_layer3[i*2] + temp_sum_layer3[i*2+1];
        end
        temp_sum_total = temp_sum_layer4[0] + temp_sum_layer4[1] + temp_sum_layer4[2] + mul_ans[48];
    end
    
    hw_sigmoid #(
        .BUS_WIDTH(BUS_WIDTH),
        .NUM_DECIMAL_IN_BINARY(NUM_DECIMAL_IN_BINARY)
    ) sigmoid (
        .x({{(BUS_WIDTH - 26){temp_sum_total[25]}}, temp_sum_total} + {{(BUS_WIDTH - 10){conv_b[filter_dim][counter_i][counter_j][9]}}, conv_b[filter_dim][counter_i][counter_j]}),
        .ans(sigmoid_out)
    );
    
endmodule