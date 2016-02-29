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
    output                          rmst_ctrl_fixed_location,
    output                 [AW-1:0] rmst_ctrl_read_base,
    output                 [AW-1:0] rmst_ctrl_read_length,
    output                          rmst_ctrl_go,
    input                           rmst_ctrl_done,

    output                          rmst_user_read_buffer,
    input                  [DW-1:0] rmst_user_buffer_data,
    input                           rmst_user_data_available,
    
    input               param_prefetch,
    input      [AW-1:0] param_waddr,
    input         [7:0] param_length_w,

    input               param_ena,
    input      [AW-1:0] param_xaddr,
    input      [AW-1:0] param_yaddr,
    input         [8:0] param_width_in,
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

localparam      [06:00]
                S_IDLE      = 'h00_00_00_01,
                S_READ_X2   = 'h00_00_00_02,
                S_WAIT_X    = 'h00_00_00_04,
                S_READ_X    = 'h00_00_00_08,
                S_WAIT_Y    = 'h00_00_00_10,
                S_READ_Y    = 'h00_00_00_20,
                S_JUDGE     = 'h00_00_00_40;

reg                 [06:00] s_cur;
reg                 [06:00] s_nxt;
reg                [AW-1:0] read_base[0:2];
wire                        go[0:2];
reg                   [2:0] done;
reg                         read[0:2];
reg                [DW-1:0] data;

reg                   [2:0] prefetch_reg;
reg                [AW-1:0] yaddr;
reg                  [17:0] length_in;
reg                  [17:0] length_out;
reg                   [2:0] cnt_write;

reg                         select_weight;
reg                   [7:0] length_w;
reg                   [7:0] cnt_ww;
reg                   [7:0] cnt_rw;
reg                         ena_write_w;
reg                   [7:0] addr_write_w;
reg                   [7:0] addr_read_w;
reg                   [3:0] cnt_read_w;
reg          [KS*KS*32-1:0] pxl_w_reg;
wire                 [31:0] ram_wq;

reg                         select_x;
reg                   [2:0] xy_ena;
reg                   [1:0] iter_x;
reg                   [9:0] length_x;
reg                         ena_write_x;
reg                  [17:0] cnt_x;

wire                        almost_empty_x;
wire                        almost_empty_y;
wire                        empty_x;
reg                         ena_read_x;

reg                         select_y;
reg                   [1:0] iter_y;
reg                   [2:0] length_y;
reg                         ena_write_y;
reg                  [17:0] cnt_y;

assign rmst_ctrl_fixed_location = 1'b0;
assign rmst_ctrl_read_base = select_weight ? read_base[0] : (select_x ? read_base[1] : read_base[2]); 
assign rmst_ctrl_read_length = 'd2;
assign rmst_ctrl_go = go[0] | go[1] | go[2];
assign rmst_user_read_buffer = read[0] | read[1] | read[2];

always @ ( posedge clk or posedge rst ) begin
    done[0] <= rmst_ctrl_done;
    done[1] <= (~done[0]) & rmst_ctrl_done;
    done[2] <= done[1];
    prefetch_reg[0] <= param_prefetch;
    prefetch_reg[1] <= (~prefetch_reg[0]) & param_prefetch;
    prefetch_reg[2] <= prefetch_reg[1];
    if( rst == 1'b1 )   begin
        length_w <= 'b0;
        cnt_ww   <= 'd0;
    end
    else if( prefetch_reg[1] ) begin
        length_w <= param_length_w[7:2] + ((param_length_w & 8'h03) ? 8'd2 : 8'd0);
        cnt_ww   <= param_length_w[7:2] + ((param_length_w & 8'h03) ? 8'd2 : 8'd0);
    end
    else if( go[0] )
        length_w <= length_w - 8'd2;
end
always @ ( posedge clk or posedge rst )
    if( rst == 1'b1 )
        read_base[0] <= 'd0;
    else if( prefetch_reg[1] )
        read_base[0] <= param_waddr;
    else if( go[0] )
        read_base[0] <= read_base[0] + 'd8;
always @ ( posedge clk or posedge rst )
    if( rst == 1'b1 )
        select_weight <= 1'b0;
    else if( prefetch_reg[1] )
        select_weight <= 1'b1;
    else if( (cnt_rw == cnt_ww) && (!rmst_user_data_available) )
        select_weight <= 1'b0;
always @ ( posedge clk or posedge rst )
    if( rst == 1'b1 )
        cnt_rw <= 'd0;
    else if( prefetch_reg[1] )
        cnt_rw <= 'd0;
    else if( rmst_user_read_buffer )
        cnt_rw <= cnt_rw + 'd1;

assign go[0] = (prefetch_reg[2] || done[2]) && (length_w > 0);

always @ ( posedge clk )
    if( select_weight & rmst_user_data_available & (cnt_write == 'd0) )
        read[0] <= 1'b1;
    else
        read[0] <= 1'b0;
always @ ( posedge clk )
    if( rmst_user_read_buffer )
        data <= rmst_user_buffer_data;
    else if( cnt_write > 'd0 )
        data <= {32'd0, data[DW-1:32]};
always @ ( posedge clk or posedge rst )
    if( rst == 1'b1 )
        cnt_write <= 3'b0;
    else if( ((cnt_x == length_in) && select_x) || ((cnt_y == length_out) && select_y) )
        cnt_write <= 3'd0;
    else if( rmst_user_read_buffer )
        cnt_write <= 3'd4;
    else if( ena_write_w || ena_write_x || ena_write_y )
        cnt_write <= cnt_write - 3'd1;
always @ ( posedge clk )
    if( (cnt_write > 'd0) & select_weight )
        ena_write_w <= 1'b1;
    else
        ena_write_w <= 1'b0;
always @ ( posedge clk )
    if( prefetch_reg[1] )
        addr_write_w <= 8'hFF;
    else if( (cnt_write > 'd0) & select_weight )
        addr_write_w <= addr_write_w + 'd1;

always @ ( posedge clk )
    if( xy_ena[1] )
        addr_read_w <= 8'h0;
    else if( cnt_read_w > 'd0 )
        addr_read_w <= addr_read_w + 'd1;
always @ ( posedge clk )
    if( xy_ena[1] )
        cnt_read_w <= 'd9;
    else if( cnt_read_w > 'd0 )
        cnt_read_w <= cnt_read_w - 'd1;
always @ ( posedge clk )
    if( cnt_read_w > 'd0 )
        pxl_w_reg <= {ram_wq, pxl_w_reg[KS*KS*32-1:KS*(KS-1)*32]};
        
always @ ( posedge clk ) begin
    xy_ena[0] <= param_ena;
    xy_ena[1] <= (~xy_ena[0]) && param_ena;
    xy_ena[2] <= xy_ena[1];
end
always @ ( posedge clk or posedge rst ) begin
    if( rst == 1'b1 )   begin
        yaddr <= 'd0;
        length_in <= 'hff_ff_ff_ff;
        length_out <= 'hff_ff_ff_ff;
    end
    else if( xy_ena[1] ) begin
        yaddr <= param_yaddr;
        length_in <= param_length_in;
        length_out <= param_length_out;
    end
end

always @ ( posedge clk ) begin
    iter_x[1] <= iter_x[0];
    if( s_cur[2] )
        iter_x[0] <= 1'b1;
    else
        iter_x[0] <= 1'b0;
end
always @ ( posedge clk or posedge rst )
    if( rst == 1'b1 )
        length_x <= 'd0;
    else if( xy_ena[1] )
        length_x <= {2'd0, param_width_in[8:1]};
    else if( (cnt_x == length_in) && select_x )
        length_x <= 'd0;
    else if( s_cur[2] )
        length_x <= 'd4;
    else if( go[1] )
        length_x <= length_x - 'd2;
always @ ( posedge clk or posedge rst )
    if( rst == 1'b1 )
        read_base[1] <= 'd0;
    else if( xy_ena[1] )
        read_base[1] <= param_xaddr;
    else if( go[1] )
        read_base[1] <= read_base[1] + 'd8;
always @ ( posedge clk or posedge rst ) begin
    if( rst == 1'b1 )
        select_x <= 1'b0;
    else if( xy_ena[1] || s_cur[2] )
        select_x <= 1'b1;
    else if( (length_x == 'd0) && (!rmst_user_data_available) )
        select_x <= 1'b0;
end

assign go[1] = (xy_ena[2] || done[2] || iter_x[1]) && (length_x > 'd0);

always @ ( posedge clk )
    if( select_x & rmst_user_data_available & (cnt_write == 'd0) & almost_empty_x )
        read[1] <= 1'b1;
    else
        read[1] <= 1'b0;
always @ ( posedge clk )
    if( (cnt_x == length_in) && select_x )
        ena_write_x <= 1'b0;
    else if( (cnt_write > 'd0) & select_x )
        ena_write_x <= 1'b1;
    else
        ena_write_x <= 1'b0;
always @ ( posedge clk or posedge rst )
    if( xy_ena[1] | (rst == 1'b1) )
        cnt_x <= 'd0;
    else if( (cnt_x == length_in) && select_x )
        cnt_x <= cnt_x;
    else if( (cnt_write > 'd0) & select_x )
        cnt_x <= cnt_x + 'd1;

always @ ( posedge clk ) begin
    iter_y[1] <= iter_y[0];
    if( s_cur[4] )
        iter_y[0] <= 1'b1;
    else
        iter_y[0] <= 1'b0;
end
always @ ( posedge clk or posedge rst )
    if( rst == 1'b1 )
        length_y <= 'd0;
    else if( (cnt_y == length_out) && select_y )
        length_y <= 'd0;
    else if( s_cur[4] )
        length_y <= 'd4;
    else if( go[2] )
        length_y <= length_y - 'd2;
always @ ( posedge clk or posedge rst )
    if( rst == 1'b1 )
        read_base[2] <= 'd0;
    else if( xy_ena[1] )
        read_base[2] <= param_yaddr;
    else if( go[2] )
        read_base[2] <= read_base[2] + 'd8;
always @ ( posedge clk or posedge rst ) begin
    if( rst == 1'b1 )
        select_y <= 1'b0;
    else if( s_cur[4] )
        select_y <= 1'b1;
    else if( (length_y == 'd0) && (!rmst_user_data_available) )
        select_y <= 1'b0;
end

assign go[2] = (done[2] || iter_y[2]) && (length_y > 'd0);

always @ ( posedge clk )
    if( select_y & rmst_user_data_available & (cnt_write == 'd0) & almost_empty_y )
        read[2] <= 1'b1;
    else
        read[2] <= 1'b0;
always @ ( posedge clk )
    if( (cnt_y == length_out) && select_y)
        ena_write_y <= 1'b0;
    else if( (cnt_write > 'd0) & select_y )
        ena_write_y <= 1'b1;
    else
        ena_write_y <= 1'b0;
always @ ( posedge clk or posedge rst )
    if( xy_ena[1] | (rst == 1'b1) )
        cnt_y <= 'd0;
    else if( (cnt_y == length_out) && select_y)
        cnt_y <= cnt_y;
    else if( (cnt_write > 'd0) & select_y )
        cnt_y <= cnt_y + 'd1;
        
always @ ( posedge clk or posedge rst )
    if( rst == 1'b1 )
        s_cur <= S_IDLE;
    else
        s_cur <= s_nxt;
always @ ( * ) begin
    s_nxt = S_IDLE;
    case ( s_cur )
        S_IDLE:     begin
            s_nxt = S_IDLE;
            if( xy_ena[1] )
                s_nxt = S_READ_X2;
        end
        S_READ_X2:  begin
            s_nxt = S_READ_X2;
            if( ~select_x )
                s_nxt = S_WAIT_X;
        end
        S_WAIT_X:   begin
            s_nxt = S_READ_X;
            if( cnt_x == length_in )
                s_nxt = S_JUDGE;
        end
        S_READ_X:   begin
            s_nxt = S_READ_X;
            if( ~select_x )
                s_nxt = S_WAIT_Y;
        end
        S_WAIT_Y:   begin
            s_nxt = S_READ_Y;
            if( cnt_y == length_out )
                s_nxt = S_JUDGE;
        end
        S_READ_Y:   begin
            s_nxt = S_READ_Y;
            if( ~select_y )
                s_nxt = S_JUDGE;
        end
        S_JUDGE:    begin
            if( cnt_x < length_in )
                s_nxt = S_WAIT_X;
            else if( cnt_y < length_out )
                s_nxt = S_WAIT_Y;
            else
                s_nxt = S_IDLE;
        end
        default: s_nxt = S_IDLE;
    endcase
end

altsyncram	WRAM (
			.address_a (addr_write_w),
			.address_b (addr_read_w),
			.clock0 (clk),
			.data_a (data[31:0]),
			.wren_a (ena_write_w),
			.q_b (ram_wq),
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
        ena_read_x <= 1'b0;
    else if( ~empty_x )
        ena_read_x <= 1'b1;
    else
        ena_read_x <= 1'b0;

scfifo	SCFFX (
    .aclr           ( rst ),
    .clock          ( clk ),
    .data           ( data[31:0] ),
    .rdreq          ( ena_read_x ),
    .sclr           ( 1'b0 ),
    .wrreq          ( ena_write_x ),
    .almost_empty   ( almost_empty_x ),
    .almost_full    ( ),
    .empty          ( empty_x ),
    .full           ( ),
    .q              ( pxl_x ),
    .usedw          ( ),
    .eccstatus ());
defparam
    SCFFX.add_ram_output_register = "OFF",
    SCFFX.almost_empty_value = 240,
    SCFFX.almost_full_value = 250,
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

scfifo	SCFFY (
    .aclr           ( rst ),
    .clock          ( clk ),
    .data           ( data[31:0] ),
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

