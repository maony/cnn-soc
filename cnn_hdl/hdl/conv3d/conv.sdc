
set_time_format -unit ns -decimal_places 3

create_clock -name {CLOCK_IN} -period 6 -waveform { 0.000 3.00 } [get_ports {clk}]
