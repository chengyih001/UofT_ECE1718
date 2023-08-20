`timescale 1ns / 1ps

module dense_layer_1 #(parameter BUS_WIDTH = 32,
                       parameter NUM_DECIMAL_IN_BINARY = 6)  (
        input logic clk,
        input logic rst_n,
        input logic start,
        input logic signed [9:0] in_bus[5][14][14],
        output logic [16:0] ram_dense_w_rd_addr,
        input logic signed [9:0] ram_dense_w_rd_value,
        input logic signed [9:0] dense_b[120],
        output logic done,
        output logic signed [9:0] out_bus[120]
    );
    logic [9:0] load_counter, load_counter_d, load_counter_dd;
    
    logic signed [9:0] dense_w[980], sigmoid_out;
    
    logic signed [11:0] dense_layer[120];
    
    enum {IDLE, LOAD_DATA_LELAY, LOAD_DATA_LELAY2, LOAD_DATA, LOAD_DATA_AFTER, LOAD_DATA_AFTER2, COMPUTING, SIGMOID, DONE} state;
    logic [4:0] counter_j, counter_k;
    logic [7:0] counter_i, counter_l;
    logic [7:0]counter_m;
    logic signed [11:0] mul_ans[13:0], temp_sum;
//    logic signed [BUS_WIDTH-1:0] pp_0[6:0];
//    logic signed [BUS_WIDTH-1:0] pp_1[3:0];
//    logic signed [BUS_WIDTH-1:0] pp_2[1:0];
//    logic signed [BUS_WIDTH-1:0] pp_3;
    
    assign ram_dense_w_rd_addr = load_counter * 120 + counter_l;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
//            load_counter_d <= load_counter;
//            load_counter_dd <= load_counter_d;
            load_counter_dd <= 10'b0;
            done <= 1'b0;
            counter_i <= 8'b0;
            counter_j <= 5'b0;
            counter_k <= 5'b0;
            counter_l <= 8'b0;
            counter_m <= 8'b0;
            load_counter <= 10'b0;
            for (int i=0; i < 120; i++) begin
                dense_layer[i] <= {(BUS_WIDTH-1){1'b0}};
                out_bus[i] <= {(BUS_WIDTH-1){1'b0}};
            end
            for (int i=0; i < 980; i++) begin
                dense_w[i] <= {(BUS_WIDTH-1){1'b0}};
            end
        end
        
        else begin
//            load_counter_d <= load_counter;
//            load_counter_dd <= load_counter_d;
            load_counter_dd <= load_counter;
            case(state)
                IDLE: begin
                    `ifdef NO_IDLE_RESET
                    done <= 1'b0;
                    counter_i <= 8'b0;
                    counter_j <= 5'b0;
                    counter_k <= 5'b0;
                    counter_l <= 8'b0;
                    counter_m <= 8'b0;
                    load_counter <= 10'b0;
                    for (int i=0; i < 120; i++) begin
                        dense_layer[i] <= {(BUS_WIDTH-1){1'b0}};
                        out_bus[i] <= {(BUS_WIDTH-1){1'b0}};
                    end
                    for (int i=0; i < 980; i++) begin
                        dense_w[i] <= {(BUS_WIDTH-1){1'b0}};
                    end
                    `endif
                    if (start == 1'b1) begin
                        state <= LOAD_DATA_LELAY;
                    end
                end
                LOAD_DATA_LELAY: begin
                    load_counter <= load_counter + 1;
                    state <= LOAD_DATA_LELAY2;
                end
                LOAD_DATA_LELAY2: begin
                    load_counter <= load_counter + 1;
                    state <= LOAD_DATA;
                end
                LOAD_DATA: begin
                    if (load_counter < 979) begin
                        dense_w[load_counter_dd] <= ram_dense_w_rd_value;
                        load_counter <= load_counter + 1;
                    end else begin
                        load_counter <= 10'b0;
                        dense_w[load_counter_dd] <= ram_dense_w_rd_value;
                        state <= LOAD_DATA_AFTER;
                    end
                end
                LOAD_DATA_AFTER: begin
                    dense_w[load_counter_dd] <= ram_dense_w_rd_value;
                    state <= LOAD_DATA_AFTER2;
                    load_counter <= 10'b0;
                end
                LOAD_DATA_AFTER2: begin
                    dense_w[load_counter_dd] <= ram_dense_w_rd_value;
                    state <= COMPUTING;
                    load_counter <= 10'b0;
                end
                COMPUTING: begin
//                    for (int i=0; i<120; i++) {
//                        for (int j=0; j<5; j++) {
//                            for (int k=0; k<14; k++) {
//                                for (int l=0; l<14; l++) {
//                                    dense_sum[i] += dense_w[j*196+k*14+l][i] * dense_input[j][k][l];
//                                }
//                            }
//                        }
//                        dense_sum[i] += dense_b[i];
//                    }
//                    for (int i=0; i < 7; i++) begin
//                        pp_0[i] <= mul_ans[2*i] + mul_ans[2*i+1];
//                    end
//                    for (int i=0; i < 3; i++) begin
//                        pp_1[i] <= pp_0[2*i] + pp_0[2*i+1];
//                    end
//                    pp_1[3] <= pp_0[6];
//                    for (int i=0; i < 2; i++) begin
//                        pp_2[i] <= pp_1[2*i] + pp_1[2*i+1];
//                    end
//                    pp_3 <= pp_2[0] + pp_2[1];
                    
                    
//                    if (counter_m >= 0) begin
//                        dense_layer[counter_l] <= dense_layer[counter_l] + pp_3;
//                    end
//                    else begin
//                        dense_layer[counter_l] <= dense_layer[counter_l];
//                    end
                    dense_layer[counter_l] <= dense_layer[counter_l] + temp_sum;
                    
                    counter_m <= counter_m + 1;
                    
                    if (counter_k >= 13) begin
                        counter_k <= 0;
                        if (counter_j >= 4) begin
                            counter_j <= 0;
                            if (counter_i < 119) begin
                                counter_i <= counter_i + 1;
                                state <= LOAD_DATA_LELAY;
                            end
                        end
                        else begin
                            counter_j <= counter_j + 1;
                        end
                    end
                    else begin
                        counter_k <= counter_k + 1;
                    end
                    
                                        

                    if (counter_m == 69) begin
                        counter_l <= counter_l + 1;
                        counter_m <= 0;
                    end
                    
                    if (counter_l > 119) begin
                        counter_l <= 0;
                        state <= SIGMOID;
                    end
                    
                end
                SIGMOID: begin
                    if (counter_l > 119) begin
                        state <= DONE;
                    end else begin
                        counter_l <= counter_l + 1;
                        out_bus[counter_l] <= sigmoid_out;
                    end
                end
                DONE: begin
                    done <= 1'b1;
                    if (start == 1'b0)
                        state <= IDLE;
                end
                
                default: state <= IDLE;
                
            endcase
        end
    end
    
    genvar c_i;
    generate
        for (c_i=0; c_i<14; c_i++) begin
            fixed_point_mul #(
                .BUS_WIDTH(12),
                .NUM_DECIMAL_IN_BINARY(NUM_DECIMAL_IN_BINARY)
            ) mul0 (
                .a({{(2){in_bus[counter_j][counter_k][c_i][9]}}, in_bus[counter_j][counter_k][c_i]}),
                .b({{(2){dense_w[counter_j*196 + counter_k*14 + c_i][9]}}, dense_w[counter_j*196 + counter_k*14 + c_i]}),
                .ans(mul_ans[c_i])
            );
        end
    endgenerate
    
    sum_even #(
        .BUS_WIDTH(12),
        .NUM_ELEMENTS(14)
    ) mul_sum (
        .input_signal(mul_ans),
        .sum(temp_sum)
    );
    
    hw_sigmoid #(
        .BUS_WIDTH(BUS_WIDTH),
        .NUM_DECIMAL_IN_BINARY(NUM_DECIMAL_IN_BINARY)
    ) sigmoid0 (
        .x({{(BUS_WIDTH - 12){dense_layer[counter_l][11]}}, dense_layer[counter_l]} + {{(BUS_WIDTH - 10){dense_b[counter_l][9]}}, dense_b[counter_l]}),
        .ans(sigmoid_out)
    );
    
endmodule