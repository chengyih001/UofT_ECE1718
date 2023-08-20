`timescale 1ns / 1ps

module hw_exp #(parameter BUS_WIDTH = 32,
                parameter NUM_DECIMAL_IN_BINARY = 6) (
        input logic signed [BUS_WIDTH-1:0] x,
        output logic signed [BUS_WIDTH-1:0] ans
    );
    
    logic signed [BUS_WIDTH-1:0] x_sq, mul1_op1, mul1_op2, mul1_ans, mul2_op1, mul2_op2, mul2_ans;
    
    always_comb begin
        mul1_op1 = 32'hffffffff;
        mul1_op2 = 32'hffffffff;
        mul2_op1 = 32'hffffffff;
        mul2_op2 = 32'hffffffff;
        if (x < -608) begin // if x is negtive and less then -608 (use '>' due to 2's compliment)
            mul1_op1 = 0;
            mul1_op2 = x;
            ans = mul1_ans + 0;
        end else if (-608 <= x && x < -576) begin
            mul1_op1 = 0;
            mul1_op2 = x;
        end else if (-576 <= x && x < -544) begin
            mul1_op1 = 0;
            mul1_op2 = x;
            mul2_op1 = 0;
            mul2_op2 = x_sq;
            ans = mul1_ans + mul2_ans + 0;       
        end else if (-544 <= x && x < -512) begin
            mul1_op1 = 0;
            mul1_op2 = x;
            mul2_op1 = 0;
            mul2_op2 = x_sq;
            ans = mul1_ans + mul2_ans + 1;       
        end else if (-512 <= x && x < -480) begin
            mul1_op1 = 0;
            mul1_op2 = x;
            mul2_op1 = 0;
            mul2_op2 = x_sq;
            ans = mul1_ans + mul2_ans + 1;
        end else if (-480 <= x && x < -448) begin
            mul1_op1 = 0;
            mul1_op2 = x;
            mul2_op1 = 0;
            mul2_op2 = x_sq;
            ans = mul1_ans + mul2_ans + 2;
        end else if (-448 <= x && x < -416) begin
            mul1_op1 = 1;
            mul1_op2 = x;
            mul2_op1 = 0;
            mul2_op2 = x_sq;
            ans = mul1_ans + mul2_ans + 2;
        end else if (-416 <= x && x < -384) begin
            mul1_op1 = 1;
            mul1_op2 = x;
            mul2_op1 = 0;
            mul2_op2 = x_sq;
            ans = mul1_ans + mul2_ans + 3;
        end else if (-384 <= x && x < -352) begin
            mul1_op1 = 1;
            mul1_op2 = x;
            mul2_op1 = 0;
            mul2_op2 = x_sq;
            ans = mul1_ans + mul2_ans + 5;
        end else if (-352 <= x && x < -320) begin
            mul1_op1 = 2;
            mul1_op2 = x;
            mul2_op1 = 0;
            mul2_op2 = x_sq;
            ans = mul1_ans + mul2_ans + 7;
        end else if (-320 <= x && x < -288) begin
            mul1_op1 = 3;
            mul1_op2 = x;
            mul2_op1 = 0;
            mul2_op2 = x_sq;
            ans = mul1_ans + mul2_ans + 9;
        end else if (-288 <= x && x < -256) begin
            mul1_op1 = 5;
            mul1_op2 = x;
            mul2_op1 = 0;
            mul2_op2 = x_sq;
            ans = mul1_ans + mul2_ans + 13;
        end else if (-256 <= x && x < -224) begin
            mul1_op1 = 7;
            mul1_op2 = x;
            mul2_op1 = 1;
            mul2_op2 = x_sq;
            ans = mul1_ans + mul2_ans + 18;
        end else if (-224 <= x && x < -192) begin
            mul1_op1 = 11;
            mul1_op2 = x;
            mul2_op1 = 1;
            mul2_op2 = x_sq;
            ans = mul1_ans + mul2_ans + 24;
        end else if (-192 <= x && x < -160) begin
            mul1_op1 = 15;
            mul1_op2 = x;
            mul2_op1 = 2;
            mul2_op2 = x_sq;
            ans = mul1_ans + mul2_ans + 31;
        end else if (-160 <= x && x < -128) begin
            mul1_op1 = 22;
            mul1_op2 = x;
            mul2_op1 = 3;
            mul2_op2 = x_sq;
            ans = mul1_ans + mul2_ans + 39;
        end else if (-128 <= x && x < -96) begin
            mul1_op1 = 31;
            mul1_op2 = x;
            mul2_op1 = 6;
            mul2_op2 = x_sq;
            ans = mul1_ans + mul2_ans + 48;
        end else if (-96 <= x && x < -64) begin
            mul1_op1 = 41;
            mul1_op2 = x;
            mul2_op1 = 9;
            mul2_op2 = x_sq;
            ans = mul1_ans + mul2_ans + 56;
        end else if (-64 <= x && x < -32) begin
            mul1_op1 = 53;
            mul1_op2 = x;
            mul2_op1 = 15;
            mul2_op2 = x_sq;
            ans = mul1_ans + mul2_ans + 62;
        end else if (-32 <= x && x < 0) begin
            mul1_op1 = 63;
            mul1_op2 = x;
            mul2_op1 = 25;
            mul2_op2 = x_sq;
            ans = mul1_ans + mul2_ans + 64;
        end else if (0 <= x && x < 32) begin
            mul1_op1 = 62;
            mul1_op2 = x;
            mul2_op1 = 41;
            mul2_op2 = x_sq;
            ans = mul1_ans + mul2_ans + 64;
        end else if (32 <= x && x < 64) begin
            mul1_op1 = 35;
            mul1_op2 = x;
            mul2_op1 = 68;
            mul2_op2 = x_sq;
            ans = mul1_ans + mul2_ans + 71;
        end else if (64 <= x && x < 96) begin
            mul1_op1 = -54;
            mul1_op2 = x;
            mul2_op1 = 112;
            mul2_op2 = x_sq;
            ans = mul1_ans + mul2_ans + 117;
        end else if (96 <= x && x < 128) begin
            mul1_op1 = -274;
            mul1_op2 = x;
            mul2_op1 = 184;
            mul2_op2 = x_sq;
            ans = mul1_ans + mul2_ans + 283;
        end else if (128 <= x && x < 160) begin
            mul1_op1 = -754;
            mul1_op2 = x;
            mul2_op1 = 303;
            mul2_op2 = x_sq;
            ans = mul1_ans + mul2_ans + 769;
        end else if (160 <= x && x < 192) begin
            mul1_op1 = -1744;
            mul1_op2 = x;
            mul2_op1 = 500;
            mul2_op2 = x_sq;
            ans = mul1_ans + mul2_ans + 2014;
        end else if (192 <= x && x < 224) begin
            mul1_op1 = -3701;
            mul1_op2 = x;
            mul2_op1 = 825;
            mul2_op2 = x_sq;
            ans = mul1_ans + mul2_ans + 4965;
        end else if (224 <= x && x < 256) begin
            mul1_op1 = -7461;
            mul1_op2 = x;
            mul2_op1 = 1360;
            mul2_op2 = x_sq;
            ans = mul1_ans + mul2_ans + 11577;
        end else if (256 <= x && x < 288) begin
            mul1_op1 = -14544;
            mul1_op2 = x;
            mul2_op1 = 2242;
            mul2_op2 = x_sq;
            ans = mul1_ans + mul2_ans + 25798;
        end else if (288 <= x && x < 320) begin
            mul1_op1 = -27675;
            mul1_op2 = x;
            mul2_op1 = 3697;
            mul2_op2 = x_sq;
            ans = mul1_ans + mul2_ans + 55447;
        end else if (320 <= x && x < 352) begin
            mul1_op1 = -51723;
            mul1_op2 = x;
            mul2_op1 = 6095;
            mul2_op2 = x_sq;
            ans = mul1_ans + mul2_ans + 115755;
        end else if (352 <= x && x < 384) begin
            mul1_op1 = -95325;
            mul1_op2 = x;
            mul2_op1 = 10049;
            mul2_op2 = x_sq;
            ans = mul1_ans + mul2_ans + 235998;
        end else if (384 <= x && x < 416) begin
            mul1_op1 = -173733;
            mul1_op2 = x;
            mul2_op1 = 16567;
            mul2_op2 = x_sq;
            ans = mul1_ans + mul2_ans + 471819;
        end else if (416 <= x && x < 448) begin
            mul1_op1 = -313752;
            mul1_op2 = x;
            mul2_op1 = 27315;
            mul2_op2 = x_sq;
            ans = mul1_ans + mul2_ans + 927945;
        end else if (448 <= x && x < 480) begin
            mul1_op1 = -562323;
            mul1_op2 = x;
            mul2_op1 = 45035;
            mul2_op2 = x_sq;
            ans = mul1_ans + mul2_ans + 1799825;
        end else if (480 <= x && x < 512) begin
            mul1_op1 = -1001364;
            mul1_op2 = x;
            mul2_op1 = 74250;
            mul2_op2 = x_sq;
            ans = mul1_ans + mul2_ans + 3449529;
        end else if (512 <= x && x < 544) begin
            mul1_op1 = -1773389;
            mul1_op2 = x;
            mul2_op1 = 122417;
            mul2_op2 = x_sq;
            ans = mul1_ans + mul2_ans + 6543405;
        end else if (544 <= x && x < 576) begin
            mul1_op1 = -3125655;
            mul1_op2 = x;
            mul2_op1 = 201832;
            mul2_op2 = x_sq;
            ans = mul1_ans + mul2_ans + 12300617;
        end else if (576 <= x && x < 608) begin
            mul1_op1 = -5486099;
            mul1_op2 = x;
            mul2_op1 = 332765;
            mul2_op2 = x_sq;
            ans = mul1_ans + mul2_ans + 22940155;
        end else if (608 <= x) begin
            mul1_op1 = -9593687;
            mul1_op2 = x;
            mul2_op1 = 548636;
            mul2_op2 = x_sq;
            ans = mul1_ans + mul2_ans + 42481615;
        end else
            ans = 32'hffffffff;
    end
    
    fixed_point_mul #(
        .BUS_WIDTH(BUS_WIDTH),
        .NUM_DECIMAL_IN_BINARY(NUM_DECIMAL_IN_BINARY)
    ) mul1 (
        .a(mul1_op1),
        .b(mul1_op2),
        .ans(mul1_ans)
    );
    
    fixed_point_mul #(
        .BUS_WIDTH(BUS_WIDTH),
        .NUM_DECIMAL_IN_BINARY(NUM_DECIMAL_IN_BINARY)
    ) mul2 (
        .a(mul2_op1),
        .b(mul2_op2),
        .ans(mul2_ans)
    );
    
    fixed_point_mul #(
        .BUS_WIDTH(BUS_WIDTH),
        .NUM_DECIMAL_IN_BINARY(NUM_DECIMAL_IN_BINARY)
    ) mul3 (
        .a(x),
        .b(x),
        .ans(x_sq)
    );
endmodule