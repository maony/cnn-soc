/*
* Created           : mny
* Date              : 201602
*/

// synposys translate_off
`timescale 1ns/100ps
// synposys translate_on

module conv2d_rmem #(
    parameter KS = 3,
    parameter AW = 32,
    parameter DW = 128
)(
    output                          rmst0_ctrl_fixed_location,
    output                 [AW-1:0] rmst0_ctrl_read_base,
    output                 [AW-1:0] rmst0_ctrl_read_length,
    output                          rmst0_ctrl_go,
    input                           rmst0_ctrl_done,
    output                          rmst0_user_read_buffer,
    input                  [DW-1:0] rmst0_user_buffer_data,
    input                           rmst0_user_data_available,
    
    output                          rmst1_ctrl_fixed_location,
    output                 [AW-1:0] rmst1_ctrl_read_base,
    output                 [AW-1:0] rmst1_ctrl_read_length,
    output                          rmst1_ctrl_go,
    input                           rmst1_ctrl_done,
    output                          rmst1_user_read_buffer,
    input                  [DW-1:0] rmst1_user_buffer_data,
    input                           rmst1_user_data_available,
    
    input               param_prefetch,
    input      [AW-1:0] param_waddr,
    input         [7:0] param_length_w,

    input               param_ena,
    input      [AW-1:0] param_xaddr,
    input      [AW-1:0] param_yaddr,
    input        [17:0] param_length_in,
    input        [17:0] param_length_out,

    output   [KS*KS*32-1:0] pxl_w,
    output                  pxl_ena_x,
    output           [31:0] pxl_x,
    input                   pxl_ena_y,
    output           [31:0] pxl_y,

    input           rst,
    input           clk
);

reg                [AW-1:0] read_base[0:2];
wire                        go[0:2];
reg                   [2:0] done[0:1];
reg                         read[0:2];
reg                [DW-1:0] data_read[0:1];
reg                   [6:0] read_length[0:2];

reg                   [3:0] prefetch;
reg                   [3:0] run_ena;

reg                   [7:0] length_w;
reg                         select_weight;
reg                  [31:0] data_shift[0:1];
reg                   [2:0] cnt_word[0:1];
reg                         ena_write_w;
reg                   [7:0] addr_write_w;
reg                   [7:0] addr_read_w;
reg                   [3:0] cnt_read_w;
reg                [DW-1:0] pxl_w_reg;

reg                  [15:0] length_x;
reg                  [17:0] length_in;
reg                         ena_write_x;
reg                  [17:0] cnt_unit[0:1];

reg                  [15:0] length_y;
reg                  [17:0] length_out;
reg                         ena_write_y;

reg                  [31:0] data_wout;
reg                         syn_xy;
reg                  [17:0] cnt_read_x;
reg                         ena_read_x;
reg                         almost_empty_x;
reg                         almost_full_x;
reg                         empty_x;
reg                         almost_empty_y;

assign rmst0_ctrl_fixed_location = 1'b0;
assign rmst0_ctrl_read_base = select_weight ? read_base[0] : read_base[1];
assign rmst0_ctrl_read_length = select_weight ? read_length[0] : read_length[1];
assign rmst0_ctrl_go = go[0] | go[1];
assign rmst0_user_read_buffer = read[0] | read[1];

assign rmst1_ctrl_fixed_location = 1'b0;
assign rmst1_ctrl_read_base = read_base[2];
assign rmst1_ctrl_read_length = read_length[2];
assign rmst1_ctrl_go = go[2];
assign rmst1_user_read_buffer = read[2];

always @ ( posedge clk ) begin
    done[0][0] <= rmst0_ctrl_done;
    done[0][1] <= (~done[0][0]) & rmst0_ctrl_done;
    done[0][2] <= done[0][1];
    done[1][0] <= rmst1_ctrl_done;
    done[1][1] <= (~done[1][0]) & rmst1_ctrl_done;
    done[1][2] <= done[1][1];
    prefetch[0] <= param_prefetch;
    prefetch[1] <= (~prefetch[0]) & param_prefetch;
    prefetch[2] <= prefetch[1];
    prefetch[3] <= prefetch[2];
    run_ena[0] <= param_ena;
    run_ena[1] <= (~run_ena[0]) && param_ena;
    run_ena[2] <= run_ena[1];
    run_ena[3] <= run_ena[2];
end
always @ ( posedge clk posedge rst )
    if( rst == 1'b1 )
        length_w <= 'b0;
    else if( prefetch[1] )
        length_w <= param_length_w[7:2] + 'd1;
    else if( go[0] )
        length_w <= length_w - read_length[0];
always @ ( posedge clk posedge rst )
    if( rst == 1'b1 )
        read_length[0] <= 'd0;
    else if( length_w > 'd32 )
        read_length[0] <= 'd32;
    else if( length_w > 'd16 )
        read_length[0] <= 'd16;
    else if( length_w > 'd4 )
        read_length[0] <= 'd4;
    else
        read_length[0] <= 'd2;
always @ ( posedge clk or posedge rst )
    if( rst == 1'b1 )
        read_base[0] <= 'd0;
    else if( prefetch[1] )
        read_base[0] <= param_waddr;
    else if( go[0] )
        read_base[0] <= read_base[0] + {read_length[0], 2'b00};
always @ ( posedge clk or posedge rst )
    if( rst == 1'b1 )
        select_weight <= 1'b0;
    else if( prefetch[1] )
        select_weight <= 1'b1;
    else if( run_ena[1] )
        select_weight <= 1'b0;

assign go[0] = (prefetch[3] || done[0][2]) && (length_w > 0);

always @ ( posedge clk )
    if( select_weight & rmst0_user_data_available & (cnt_word[0] == 'd0) & (~read[0]) )
        read[0] <= 1'b1;
    else
        read[0] <= 1'b0;
always @ ( posedge clk )    begin
    data_shift[0] <= data_read[0][31:0];
    if( rmst0_user_read_buffer )
        data_read[0] <= rmst0_user_buffer_data;
    else if( cnt_word[0] > 'd0 )
        data_read[0] <= {32'd0, data_read[0][DW-1:32]};
end
always @ ( posedge clk or posedge rst )
    if( rst == 1'b1 )
        cnt_word[0] <= 3'b0;
    else if( rmst0_user_read_buffer )
        cnt_word[0] <= 3'd4;
    else if( cnt_word[0] > 'd0 )
        cnt_word[0] <= cnt_word[0] - 3'd1;        
always @ ( posedge clk )
    if( (cnt_word[0] > 'd0) & select_weight )
        ena_write_w <= 1'b1;
    else
        ena_write_w <= 1'b0;
always @ ( posedge clk )
    if( prefetch[1] )
        addr_write_w <= 8'hFF;
    else if( (cnt_word[0] > 'd0) & select_weight )
        addr_write_w <= addr_write_w + 'd1;
always @ ( posedge clk )
    if( prefetch[1] )
        addr_read_w <= 8'h0;
    else if( cnt_read_w > 'd0 )
        addr_read_w <= addr_read_w + 'd1;
always @ ( posedge clk )
    if( run_ena[1] )
        cnt_read_w <= 'd10;
    else if( cnt_read_w > 'd0 )
        cnt_read_w <= cnt_read_w - 'd1;
always @ ( posedge clk )
    if( cnt_read_w > 'd0 )
        pxl_w_reg <= {data_wout, pxl_w_reg[KS*KS*32-1:32]};
assign pxl_w = pxl_w_reg;
        
always @ ( posedge clk or posedge rst )
    if( rst == 1'b1 )       begin
        length_x    <= 'b0;
        length_in   <= 'd0;
    end
    else if( run_ena[1] )   begin
        length_x    <= param_length_in[7:2];
        length_in   <= param_length_in;
    end
    else if( go[1] )
        length_x <= length_x - read_length[1];
always @ ( posedge clk posedge rst )
    if( rst == 1'b1 )
        read_length[1] <= 'd0;
    else if( length_x > 'd32 )
        read_length[1] <= 'd32;
    else if( length_x > 'd16 )
        read_length[1] <= 'd16;
    else if( length_x > 'd4 )
        read_length[1] <= 'd4;
    else
        read_length[1] <= 'd2;
always @ ( posedge clk or posedge rst )
    if( rst == 1'b1 )
        read_base[1] <= 'd0;
    else if( run_ena[1] )
        read_base[1] <= param_xaddr;
    else if( go[1] )
        read_base[1] <= read_base[1] + {read_length[1], 2'b00};

assign go[1] = (run_ena[3]] || done[0][2]) && (length_x > 0);

always @ ( posedge clk )
    if( (~select_weight) & rmst0_user_data_available & (cnt_word[0] == 'd0) & (~read[1]) & almost_empty_x )
        read[1] <= 1'b1;
    else
        read[1] <= 1'b0;
always @ ( posedge clk or posedge rst )
    if( rst == 1'b1 )
        ena_write_x <= 1'b0;
    else if( (cnt_unit[0] == length_in) && (~select_w) )
        ena_write_x <= 1'b0;
    else if( (cnt_word[0] > 'd0) && (~select_w) )
        ena_write_x <= 1'b1;
    else
        ena_write_x <= 1'b0;
always @ ( posedge clk or posedge rst )
    if( run_ena[1] | (rst == 1'b1) )
        cnt_unit[0] <= 'd0;
    else if( (cnt_unit[0] == length_in) && (~select_w) )
        cnt_unit[0] <= cnt_unit[0];
    else if( (cnt_word[0] > 'd0) && (~select_w) )
        cnt_unit[0] <= cnt_unit[0] + 'd1;

always @ ( posedge clk or posedge rst )
    if( rst == 1'b1 )       begin
        length_y    <= 'b0;
        length_out  <= 'd0;
    end
    else if( run_ena[1] )   begin
        length_y    <= param_length_out[7:2];
        length_out  <= param_length_out;
    end
    else if( go[2] )
        length_y <= length_y - read_length[2];
always @ ( posedge clk posedge rst )
    if( rst == 1'b1 )
        read_length[2] <= 'd0;
    else if( length_y > 'd32 )
        read_length[2] <= 'd32;
    else if( length_y > 'd16 )
        read_length[2] <= 'd16;
    else if( length_y > 'd4 )
        read_length[2] <= 'd4;
    else
        read_length[2] <= 'd2;
always @ ( posedge clk or posedge rst )
    if( rst == 1'b1 )
        read_base[2] <= 'd0;
    else if( run_ena[1] )
        read_base[2] <= param_yaddr;
    else if( go[2] )
        read_base[2] <= read_base[2] + {read_length[2], 2'b00};

assign go[2] = (run_ena[3]] || done[1][2]) && (length_y > 0);

always @ ( posedge clk )
    if( rmst1_user_data_available & (cnt_word[1] == 'd0) & (~read[2]) & almost_empty_y )
        read[2] <= 1'b1;
    else
        read[2] <= 1'b0;
always @ ( posedge clk )    begin
    data_shift[1] <= data_read[1][31:0];
    if( rmst1_user_read_buffer )
        data_read[1] <= rmst1_user_buffer_data;
    else if( cnt_word[1] > 'd0 )
        data_read[1] <= {32'd0, data_read[1][DW-1:32]};
end
always @ ( posedge clk or posedge rst )
    if( rst == 1'b1 )
        cnt_word[1] <= 3'b0;
    else if( rmst1_user_read_buffer )
        cnt_word[1] <= 3'd4;
    else if( cnt_word[1] > 'd0 )
        cnt_word[1] <= cnt_word[1] - 3'd1;        
always @ ( posedge clk or posedge rst )
    if( rst == 1'b1 )
        ena_write_y <= 1'b0;
    else if( cnt_unit[1] == length_out )
        ena_write_y <= 1'b0;
    else if( cnt_word[1] > 'd0 )
        ena_write_y <= 1'b1;
    else
        ena_write_y <= 1'b0;
always @ ( posedge clk or posedge rst )
    if( run_ena[1] | (rst == 1'b1) )
        cnt_unit[1] <= 'd0;
    else if( cnt_unit[1] == length_out )
        cnt_unit[1] <= cnt_unit[1];
    else if( cnt_word[1] > 'd0 )
        cnt_unit[1] <= cnt_unit[1] + 'd1;

altsyncram	WRAM (
			.address_a (addr_write_w),
			.address_b (addr_read_w),
			.clock0 (clk),
			.data_a (data_shift[0]),
			.wren_a (ena_write_w),
			.q_b (data_wout),
			.aclr0 (1'b0),
			.aclr1 (1'b0),
			.addressstall_a (1'b0),
			.addressstall_b (1'b0),
			.byteena_a (1'b1),
			.byteena_b (1'b1),
			.clock1 (1'b1),
			.clocken0 (1'b1),
			.clocken1 (1'b1),
			.clocken2 (1'b1),
			.clocken3 (1'b1),
			.data_b ({32{1'b1}}),
			.eccstatus (),
			.q_a (),
			.rden_a (1'b1),
			.rden_b (1'b1),
			.wren_b (1'b0));
defparam
	WRAM.address_aclr_b = "NONE",
	WRAM.address_reg_b = "CLOCK0",
	WRAM.clock_enable_input_a = "BYPASS",
	WRAM.clock_enable_input_b = "BYPASS",
	WRAM.clock_enable_output_b = "BYPASS",
	WRAM.intended_device_family = "Cyclone V",
	WRAM.lpm_type = "altsyncram",
	WRAM.numwords_a = 256,
	WRAM.numwords_b = 256,
	WRAM.operation_mode = "DUAL_PORT",
	WRAM.outdata_aclr_b = "NONE",
	WRAM.outdata_reg_b = "UNREGISTERED",
	WRAM.power_up_uninitialized = "FALSE",
	WRAM.read_during_write_mode_mixed_ports = "OLD_DATA",
	WRAM.widthad_a = 8,
	WRAM.widthad_b = 8,
	WRAM.width_a = 32,
	WRAM.width_b = 32,
	WRAM.width_byteena_a = 1;

assign pxl_ena_x = ena_read_x;
always @ ( posedge clk or posedge rst )
    if( rst == 1'b1 )
        syn_xy <= 1'b0;
    else if( run_ena[1] )
        syn_xy <= 1'b0;
    else if( almost_full_x )
        syn_xy <= 1'b1;
always @ ( posedge clk or posedge rst )
    if( rst == 1'b1 )
        cnt_read_x <= 'd0;
    else if( run_ena[1] )
        cnt_read_x <= 'd0;
    else if( cnt_read_x == length_in )
        cnt_read_x <= cnt_read_x;
    else if( (~empty_x) & syn_xy )
        cnt_read_x <= cnt_read_x + 'd1;
always @ ( posedge clk or posedge rst )
    if( rst == 1'b1 )
        ena_read_x <= 1'b0;
    else if( cnt_read_x == length_in )
        ena_read_x <= 1'b0;
    else if( (~empty_x) & syn_xy )
        ena_read_x <= 1'b1;
    else
        ena_read_x <= 1'b0;

scfifo	SCFFX (
    .aclr           ( rst ),
    .clock          ( clk ),
    .data           ( data_shift[0] ),
    .rdreq          ( ena_read_x ),
    .sclr           ( 1'b0 ),
    .wrreq          ( ena_write_x ),
    .almost_empty   ( almost_empty_x ),
    .almost_full    ( almost_full_x ),
    .empty          ( empty_x ),
    .full           ( ),
    .q              ( pxl_x ),
    .usedw          ( ),
    .eccstatus ());
defparam
    SCFFX.add_ram_output_register = "OFF",
    SCFFX.almost_empty_value = 240,
    SCFFX.almost_full_value = 200,
    SCFFX.intended_device_family = "Cyclone V",
    SCFFX.lpm_hint = "RAM_BLOCK_TYPE=M10K",
    SCFFX.lpm_numwords = 256,
    SCFFX.lpm_showahead = "ON",
    SCFFX.lpm_type = "scfifo",
    SCFFX.lpm_width = 32,
    SCFFX.lpm_widthu = 8,
    SCFFX.overflow_checking = "ON",
    SCFFX.underflow_checking = "ON",
    SCFFX.use_eab = "ON";
    
    // synopsys translate_off
    integer fp_x;
    integer fp_y;
    reg         [21:0] cnt_xf;
    reg         [21:0] cnt_yf;
    initial begin
        fp_x = $fopen("x read.txt");
        cnt_xf = 0;
        
        forever begin
            @( posedge clk);
            if( ena_write_x ) begin
                $fwrite(fp_x, "%h   ", ram_wdata);
                cnt_xf = cnt_xf + 1;
                if( cnt_xf % param_width_in == 0 )
                    $fwrite(fp_x, "\n");
            end
            if( cnt_xf == param_length_in) begin
                $fclose(fp_x);
            end
        end
    end
    initial begin
        fp_y = $fopen("y read.txt");
        cnt_yf = 0;
        
        forever begin
            @( posedge clk);
            if( ena_write_y ) begin
                $fwrite(fp_y, "%h   ", ram_wdata);
                cnt_yf = cnt_yf + 1;
                if( cnt_yf % (param_width_in-2) == 0 )
                    $fwrite(fp_y, "\n");
            end
            if( cnt_yf == param_length_out) begin
                $fclose(fp_y);
            end
        end
    end
    // synopsys translate_on
scfifo	SCFFY (
    .aclr           ( rst ),
    .clock          ( clk ),
    .data           ( data_shift[1] ),
    .rdreq          ( pxl_ena_y ),
    .sclr           ( 1'b0 ),
    .wrreq          ( ena_write_y ),
    .almost_empty   ( almost_empty_y ),
    .almost_full    ( ),
    .empty          ( ),
    .full           ( ),
    .q              ( pxl_y ),
    .usedw          ( ),
    .eccstatus ());
defparam
    SCFFY.add_ram_output_register = "OFF",
    SCFFY.almost_empty_value = 240,
    SCFFY.almost_full_value = 250,
    SCFFY.intended_device_family = "Cyclone V",
    SCFFY.lpm_hint = "RAM_BLOCK_TYPE=M10K",
    SCFFY.lpm_numwords = 256,
    SCFFY.lpm_showahead = "OFF",
    SCFFY.lpm_type = "scfifo",
    SCFFY.lpm_width = 32,
    SCFFY.lpm_widthu = 8,
    SCFFY.overflow_checking = "ON",
    SCFFY.underflow_checking = "ON",
    SCFFY.use_eab = "ON";
endmodule

