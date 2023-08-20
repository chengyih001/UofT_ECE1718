#create_clock -period 3.333 -name clk [get_ports clk]
create_clock -period 20 -name clk [get_ports clk]

set_property PACKAGE_PIN K22 [get_ports clk]
set_property IOSTANDARD LVCMOS18 [get_ports clk]
