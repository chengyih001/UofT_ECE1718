`timescale 1ns / 1ps

module forward_pass_top #(parameter BUS_WIDTH = 32,
                          parameter NUM_DECIMAL_IN_BINARY = 6) (
        input logic clk,
        input logic rst_n,
        input logic [3:0] control,
        input logic signed [BUS_WIDTH-1:0] in_bus,
        output logic done,
        output logic signed [BUS_WIDTH-1:0] out_bus
    );
    // inputs
    logic signed [14:0] img[1120];
    logic signed [14:0] conv_w[5][7][7];
    logic signed [9:0] conv_b[5][28][28];
    logic signed [9:0] dense_w[980]/*[120]*/;
    logic signed [9:0] dense_b[120];
    logic signed [9:0] dense_w2[120][10];
    logic signed [BUS_WIDTH-1:0] dense_b2[10];
    // local
//    logic signed [BUS_WIDTH-1:0] conv_layer[5][28][28];
    logic signed [9:0] sig_layer[5][28][28];
    logic signed [9:0] max_layer[5][14][14];
    logic signed [9:0] dense_sigmoid[120];
    logic signed [BUS_WIDTH-1:0] dense_sum2[10];
    // outputs
    logic signed [9:0] dense_softmax[10];
    
    enum {IDLE, LOAD_IMG, LOAD_CONV_W, LOAD_CONV_B, LOAD_DENSE_W, LOAD_DENSE_B, LOAD_DENSE_W2, LOAD_DENSE_B2, CONVOLUTION, MAX_POOLING, DENSE_LAYER1, DENSE_LAYER2, 
          DATA_SCATTER, DONE} top_state;
    logic img_loader_en, conv_w_loader_en, conv_b_loader_en, dense_w_loader_en, dense_b_loader_en, dense_w2_loader_en, dense_b2_loader_en;
    logic signed [BUS_WIDTH-1:0] in_bus_d;
    
    logic [16:0] ram_dense_w_wr_addr, ram_dense_w_rd_addr;
    logic [9:0] ram_dense_w_wr_value, ram_dense_w_rd_value;
    logic ram_dense_w_wr_en;
    
    logic convolution_layer_start, convolution_layer_done, max_pooling_layer_start, max_pooling_layer_done;
    logic dense_layer1_start, dense_layer1_done, dense_layer2_start, dense_layer2_done;
    
    logic [3:0] scatter_counter;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
//            in_bus_d <= in_bus;
            in_bus_d <= {(BUS_WIDTH-1){1'b0}};
            done <= 1'b0;
            out_bus <= {(BUS_WIDTH-1){1'b0}};
            img_loader_en <= 1'b0;
            conv_w_loader_en <= 1'b0;
            conv_b_loader_en <= 1'b0;
            dense_b_loader_en <= 1'b0;
            dense_w2_loader_en <= 1'b0;
            dense_b2_loader_en <= 1'b0;
            ram_dense_w_wr_addr <= 17'b0;
            ram_dense_w_wr_value <= {(BUS_WIDTH-1){1'b0}};
            ram_dense_w_wr_en <= 1'b0;
            convolution_layer_start <= 1'b0;
            max_pooling_layer_start <= 1'b0;
            dense_layer1_start <= 1'b0;
            dense_layer2_start <= 1'b0;
            scatter_counter <= 4'b0;
//            for (int i=0; i<120; i++) begin
//                dense_w[i] <= {(BUS_WIDTH-1){1'b0}};
//            end
        end else begin
            in_bus_d <= in_bus;
            case (top_state)
                IDLE: begin
                    done <= 1'b0;
                    out_bus <= {(BUS_WIDTH-1){1'b0}};
                    img_loader_en <= 1'b0;
                    conv_w_loader_en <= 1'b0;
                    conv_b_loader_en <= 1'b0;
                    dense_b_loader_en <= 1'b0;
                    dense_w2_loader_en <= 1'b0;
                    dense_b2_loader_en <= 1'b0;
                    ram_dense_w_wr_addr <= 17'b0;
                    ram_dense_w_wr_value <= {(BUS_WIDTH-1){1'b0}};
                    ram_dense_w_wr_en <= 1'b0;
                    convolution_layer_start <= 1'b0;
                    max_pooling_layer_start <= 1'b0;
                    dense_layer1_start <= 1'b0;
                    dense_layer2_start <= 1'b0;
                    scatter_counter <= 4'b0;
//                    for (int i=0; i<120; i++) begin
//                        dense_w[i] <= {(BUS_WIDTH-1){1'b0}};
//                    end
                    
                    if (control == 4'd1) begin
                        img_loader_en <= 1'b1;
                        top_state <= LOAD_IMG;
                    end
                end
                LOAD_IMG: begin
                    if (control == 4'd1)
                        img_loader_en <= 1'b1;
                    else if (control == 4'd2) begin 
                        img_loader_en <= 1'b0;
                        conv_w_loader_en <= 1'b1;
                        top_state <= LOAD_CONV_W;
                    end
                end
                LOAD_CONV_W: begin
                    if (control == 4'd2)
                        conv_w_loader_en <= 1'b1;
                    else if (control == 4'd3) begin 
                        conv_w_loader_en <= 1'b0;
                        conv_b_loader_en <= 1'b1;
                        top_state <= LOAD_CONV_B;
                    end
                end
                LOAD_CONV_B: begin
                    if (control == 4'd3)
                        conv_b_loader_en <= 1'b1;
                    else if (control == 4'd4) begin 
                        conv_b_loader_en <= 1'b0;
                        ram_dense_w_wr_en <= 1'b1;
                        ram_dense_w_wr_addr <= 17'b0;
                        ram_dense_w_wr_value <= in_bus;
//                        dense_w_loader_en <= 1'b1;
                        top_state <= LOAD_DENSE_W;
                    end
                end
                LOAD_DENSE_W: begin
                    // note that dense_w stored at ram
                    if (control == 4'd4) begin
                        ram_dense_w_wr_en <= 1'b1;
                        ram_dense_w_wr_addr <= ram_dense_w_wr_addr + 17'b1; // shoud be with in range 117600
                        ram_dense_w_wr_value <= in_bus;
                    end if (control == 4'd5) begin 
                        ram_dense_w_wr_en <= 1'b0;
                        dense_b_loader_en <= 1'b1;
                        top_state <= LOAD_DENSE_B;
                    end
//                    if (control == 4'd4)
//                        dense_w_loader_en <= 1'b1;
//                    else if (control == 4'd5) begin 
//                        dense_w_loader_en <= 1'b0;
//                        dense_b_loader_en <= 1'b1;
//                        top_state <= LOAD_DENSE_B;
//                    end
                end
                LOAD_DENSE_B: begin
                    if (control == 4'd5)
                        dense_b_loader_en <= 1'b1;
                    else if (control == 4'd6) begin 
                        dense_b_loader_en <= 1'b0;
                        dense_w2_loader_en <= 1'b1;
                        top_state <= LOAD_DENSE_W2;
                    end
                end 
                LOAD_DENSE_W2: begin
                    if (control == 4'd6)
                        dense_w2_loader_en <= 1'b1;
                    else if (control == 4'd7) begin 
                        dense_w2_loader_en <= 1'b0;
                        dense_b2_loader_en <= 1'b1;
                        top_state <= LOAD_DENSE_B2;
                    end
                end
                LOAD_DENSE_B2: begin
                    if (control == 4'd7)
                        dense_b2_loader_en <= 1'b1;
                    else if (control == 4'd0) begin 
                        dense_b2_loader_en <= 1'b0;
                        convolution_layer_start <= 1'b1;
                        top_state <= CONVOLUTION;
                    end
                end
                CONVOLUTION: begin
                    if (convolution_layer_done == 1'b1) begin
                        max_pooling_layer_start <= 1'b1;
                        top_state <= MAX_POOLING;
                    end
                end
                MAX_POOLING: begin
                    if (max_pooling_layer_done == 1'b1) begin
                        dense_layer1_start <= 1'b1;
                        top_state <= DENSE_LAYER1;
                    end
                end
                DENSE_LAYER1: begin
                    if (dense_layer1_done == 1'b1) begin
                        dense_layer2_start <= 1'b1;
                        top_state <= DENSE_LAYER2;
                    end
                end
                DENSE_LAYER2: begin
                    if (dense_layer2_done == 1'b1) begin
                        top_state <= DATA_SCATTER;
                    end
                end
                DATA_SCATTER: begin
                    done <= 1'b1;
                    if (scatter_counter >= 10) begin
                        top_state <= DONE;
                    end else begin
                        scatter_counter <= scatter_counter + 1;
                        out_bus <= {{(BUS_WIDTH-10){1'b0}}, dense_softmax[scatter_counter]};
                    end
                end
                DONE: begin
                    done <= 1'b1;
                    // goto IDLE in the next clock cycle
                    top_state <= IDLE;
                end
                default: top_state <= IDLE;
            endcase
        end
    end
    
    convolution_layer #(
        .BUS_WIDTH(BUS_WIDTH),
        .NUM_DECIMAL_IN_BINARY(NUM_DECIMAL_IN_BINARY)
    ) convolution (
        .clk(clk),
        .rst_n(rst_n),
        .start(convolution_layer_start),
        .img(img),
        .conv_w(conv_w),
        .conv_b(conv_b),
        .sig_layer(sig_layer),
        .done(convolution_layer_done)
    );
    
    max_pooling_layer #(
        .BUS_WIDTH(BUS_WIDTH),
        .NUM_DECIMAL_IN_BINARY(NUM_DECIMAL_IN_BINARY)
    ) max_pooling (
        .clk(clk),
        .rst_n(rst_n),
        .start(max_pooling_layer_start),
        .sig_layer(sig_layer),
        .max_layer(max_layer),
        .done(max_pooling_layer_done)
    );
    
    dense_layer_1 #(
        .BUS_WIDTH(BUS_WIDTH),
        .NUM_DECIMAL_IN_BINARY(NUM_DECIMAL_IN_BINARY)
    ) dense1 (
        .clk(clk),
        .rst_n(rst_n),
        .start(dense_layer1_start),
        .in_bus(max_layer),
        .ram_dense_w_rd_addr(ram_dense_w_rd_addr),
        .ram_dense_w_rd_value(ram_dense_w_rd_value),
        .dense_b(dense_b),
        .done(dense_layer1_done),
        .out_bus(dense_sigmoid)
    );
    
    dense_layer_2 #(
        .BUS_WIDTH(BUS_WIDTH),
        .NUM_DECIMAL_IN_BINARY(NUM_DECIMAL_IN_BINARY)
    ) dense2 (
        .clk(clk),
        .rst_n(rst_n),
        .start(dense_layer2_start),
        .in_bus(dense_sigmoid),
        .dense_w(dense_w2),
        .dense_b(dense_b2),
        .done(dense_layer2_done),
        .out_bus(dense_sum2)
    );
    
    softmax_layer #(
        .BUS_WIDTH(BUS_WIDTH),
        .NUM_DECIMAL_IN_BINARY(NUM_DECIMAL_IN_BINARY)
    ) softmax (
        .dense_sum2(dense_sum2),
        .dense_softmax(dense_softmax)
    );
    
    // =========================== data modules =================================================
    data_loader_1d #(
        .BUS_WIDTH(15),
        .WIDTH(1120)
    ) img_loader (
        .clk(clk),
        .rst_n(rst_n),
        .load(img_loader_en),
        .in_bus(in_bus_d),
        .signal_out(img)
    );
    
    data_loader_3d #(
        .BUS_WIDTH(15),
        .DEPTH(5),
        .HEIGHT(7),
        .WIDTH(7)
    ) conv_w_loader (
        .clk(clk),
        .rst_n(rst_n),
        .load(conv_w_loader_en),
        .in_bus(in_bus_d),
        .signal_out(conv_w)
    );
    
//    data_loader_2d #(
//        .BUS_WIDTH(BUS_WIDTH),
//        .HEIGHT(980),
//        .WIDTH(120)
//    ) dense_w_loader (
//        .clk(clk),
//        .rst_n(rst_n),
//        .load(dense_w_loader_en),
//        .in_bus(in_bus_d),
//        .signal_out(dense_w)
//    );
    
    data_loader_3d #(
        .BUS_WIDTH(10),
        .DEPTH(5),
        .HEIGHT(28),
        .WIDTH(28)
    ) conv_b_loader (
        .clk(clk),
        .rst_n(rst_n),
        .load(conv_b_loader_en),
        .in_bus(in_bus_d),
        .signal_out(conv_b)
    );
    
    data_loader_1d #(
        .BUS_WIDTH(10),
        .WIDTH(120)
    ) dense_b_loader (
        .clk(clk),
        .rst_n(rst_n),
        .load(dense_b_loader_en),
        .in_bus(in_bus_d),
        .signal_out(dense_b)
    );
    
    data_loader_2d #(
        .BUS_WIDTH(10),
        .HEIGHT(120),
        .WIDTH(10)
    ) dense_w2_loader (
        .clk(clk),
        .rst_n(rst_n),
        .load(dense_w2_loader_en),
        .in_bus(in_bus_d),
        .signal_out(dense_w2)
    );
    
    data_loader_1d #(
        .BUS_WIDTH(BUS_WIDTH),
        .WIDTH(10)
    ) dense_b2_loader (
        .clk(clk),
        .rst_n(rst_n),
        .load(dense_b2_loader_en),
        .in_bus(in_bus_d),
        .signal_out(dense_b2)
    );
    
    blk_mem_gen_0 rem_dense_w(
        .addra(ram_dense_w_wr_en ? ram_dense_w_wr_addr : ram_dense_w_rd_addr),
        .clka(clk),
        .dina(ram_dense_w_wr_value),
        .douta(ram_dense_w_rd_value),
        .ena(1'b1),
        .wea(ram_dense_w_wr_en)
    );
    
endmodule
