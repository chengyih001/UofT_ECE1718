`timescale 1ns / 1ps
`define NO_IDLE_RESET // to speed up the simulation

module tb();
    parameter BUS_WIDTH = 32; 
    parameter CLK_PERIOD = 6;
    
    logic clk, rst_n;
    integer data_file, scan_data_file, expected_file, scan_expected_file;
    
    initial clk = 1'b1;
    always #(CLK_PERIOD/2) clk <= ~clk; 
    
    initial begin 
        rst_n <= 1'b0;
        #1;
        rst_n <= 1'b1;
    end 
    
    logic done;
    logic [3:0] control;
    logic signed [BUS_WIDTH-1:0] in_bus, out_bus;
    
    // instentiation of the Unit Under Test
    forward_pass_top #(
        .BUS_WIDTH(32),
        .NUM_DECIMAL_IN_BINARY(6)
    )uut(
        .clk(clk),
        .rst_n(rst_n),
        .control(control),
        .in_bus(in_bus),
        .done(done),
        .out_bus(out_bus)
    );
    
    integer array_index;
    string temp_line;
    logic signed [BUS_WIDTH-1:0] exp_dense_softmax[10];
    
    initial begin
        data_file = $fopen("golden_vectors_in.txt", "r");
        if (data_file == 0) begin
            $display("data_file handle was 0");
            $finish;
        end
        expected_file = $fopen("golden_vectors_out.txt", "r");
        if (expected_file == 0) begin
            $display("expected_file handle was 0");
            $finish;
        end
        
        // first read the expected values for dense_softmax
        array_index = 0;
        // $fgets(temp_line, expected_file); // ignore the first line of the file
        while(!$feof(expected_file)) begin
            scan_expected_file = $fscanf(expected_file, "done = 1, val = %d\n", exp_dense_softmax[array_index]);
            array_index = array_index + 1;
        end
        $fclose(expected_file);
        
        // then read the input signals and their control values
        wait(!rst_n);
        // $fgets(temp_line, data_file); // ignore the first line of the file
        while(!$feof(data_file)) begin
            @(posedge clk);
            #0;
            scan_data_file = $fscanf(data_file, "control = %d, val = %d\n", control, in_bus);
        end
        $fclose(data_file);
        
        // set the control to 0 to start the calculation
        @(posedge clk);
        control <= 4'b0;
        
        wait(done == 1'b1);
        #0.001;
        // checking the output
        for (int i=0; i<10; i++) begin
            assert (out_bus == exp_dense_softmax[i])
                else $error("dense_softmax[%.0d] value mismatch! exp = %.0d, act = %.0d", i, exp_dense_softmax[i], out_bus);
            @(posedge clk);
            #0.001;
        end
        $finish;
    end
    
endmodule
