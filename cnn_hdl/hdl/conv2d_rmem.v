/*
* Created           : mny
* Date              : 201602
*/

// synposys translate_off
`timescale 1ns/100ps
// synposys translate_on

module conv2d_rmem #(
    parameter KS = 3,
    parameter AW = 30,
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
    input               param_width_in,
    input        [17:0] param_length_in,
    input        [17:0] param_length_out,

    output                  pxl_ena_w,
    output   [KS*KS*32-1:0] pxl_w,
    output                  pxl_ena_x,
    output           [31:0] pxl_x,
    input                   pxl_ena_y,
    output           [31:0] pxl_y,

    input           rst,
    input           clk
);

localparam      [05:00]
                W_IDLE = 'h00_00_00_01,
                W_OVER = 'h00_00_00_00;

reg                [AW-1:0] read_base[0:1];
reg                [AW-1:0] read_length[0:1];
reg                         go[0:1];
reg                         done;
reg                         read[0:1];
reg                         select_weight;

reg                   [1:0] prefetch_reg;
reg                   [7:0] length_w;
reg                   [7:0] cnt_w;

assign rmst_ctrl_fixed_location = 1'b0;
assign rmst_ctrl_read_base = select_weight ? read_base[0] : read_base[1]; 
assign rmst_ctrl_read_length = 'd2;
assign rmst_ctrl_go = select_weight ? go[0] : go[1];
assign rmst_user_read_buffer = select_weight ? read[0] : read[1];

always @ ( posedge clk or posedge rst ) begin
    done[0] <= rmst_ctrl_done;
    done[1] <= (~done[0]) & rmst_ctrl_done;
    prefetch_reg[0] <= param_prefetch;
    prefetch_reg[1] <= (~prefetch_reg[0]) & param_prefetch;
    if( rst == 1'b1 )
        length_w <= 'b0;
    else if( (~prefetch_reg[0]) & param_prefetch )
        length_w <= param_length_w[7:2] + ((param_length_w & 8'h03) ? 8'd2 : 8'd0 );
    else if( go[0] )
        length_w <= length_w - 8'd2;
end
always @ ( posedge clk )
    if( (~prefetch_reg[0]) & param_prefetch )
        read_base[0] <= param_waddr;
    else if( go[0] )
        read_base[0] <= read_base[0] + 'd8;
 
always @ ( posedge clk or posedge rst )
    if( rst == 1'b1 )
        select_weight <= 1'b0;
    else if( (~prefetch_reg[0]) & param_prefetch )
        select_weight <= 1'b1;
    else if( (length_w == 'd0) && (!rmst_user_data_available) )
        select_weight <= 1'b0;

assign go[0] = (prefetch_reg[1] || done[1]) && (length_w > 0);

always @ ( posedge clk )
    if( select_weight & rmst_user_data_available & (cnt_write == 'd0) )
        read[0] <= 1'b1;
    else
        read[0] <= 1'b0;
always @ ( posedge clk )
    if( rmst_user_read_buffer )
        data <= rmst_user_buffer_data;
    else if( cnt_write_w > 'd0 )
        data <= {32'd0, data[DW-1:32]};
always @ ( posedge clk or posedge rst )
    if( rst == 1'b1 )
        cnt_write_w <= 3'b0;
    else if( rmst_user_read_buffer )
        cnt_write_w <= 3'd4;
    else if( ena_write_w )
        cnt_write_w <= cnt_write_w - 3'd1;
always @ ( posedge clk )
    if( (cnt_write_w > 'd0) & select_weight )
        ena_write_w <= 1'b1;
    else
        ena_write_w <= 1'b0;
always @ ( posedge clk )
    if( (~prefetch_reg[0]) & param_prefetch )
        addr_write_w <= 8'hFF;
    else if( (cnt_write_w > 'd0) & select_weight )
        addr_write_w <= addr_write_w + 'd1;

always @ ( posedge clk ) begin
    xy_ena[0] <= param_ena;
    xy_ena[1] <= (~xy_ena[0]) && param_ena;
    if( (~xy_ena[0]) && param_ena ) begin
        yaddr <= param_yaddr;
        length_in <= parma_length_in;
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
always @ ( posedge clk )
    if( (~xy_ena[0]) && param_ena )
        length_x <= {1'b0, width_in[8:1]};
    else if( cnt_x >= length_in )
        length_x <= 'd0;
    else if( s_cur[2] )
        length_x <= 'd4;
    else if( go[1] ) // prefetch
        length_x <= length_x - 'd2;
always @ ( posedge clk )
    if( (~xy_ena[0]) && param_ena )
        read_base[1] <= param_xaddr;
    else if( go[1] )
        read_base[1] <= read_base[1] + 'd8;
always @ ( posedge clk or posedge rst ) begin
    if( rst == 1'b1 )
        select_x <= 1'b0;
    else if( ((~xy_ena[0]) && param_ena) || s_cur[2] )
        select_x <= 1'b1;
    else if( (length_x == 'd0) && (!rmst_user_data_available) )
        select_x <= 1'b0;
end

assign go[1] = (xy_ena[1] || done[1] || iter_x[1]) && (length_x > 'd0);

always @ ( posedge clk )
    if( select_x & rmst_user_data_available & (cnt_write == 'd0) )
        read[1] <= 1'b1;
    else
        read[1] <= 1'b0;
always @ ( posedge clk )
    if( (cnt_write_w > 'd0) & select_x )
        ena_write_x <= 1'b1;
    else
        ena_write_x <= 1'b0;
always @ ( posedge clk )
    if( (~xy_ena[0]) && param_ena )
        cnt_x <= 'd0;
    else if( (cnt_write_w > 'd0) & selsect_x )
        cnt_x <= cnt_x + 'd1;

always @ ( posedge clk or posedge rst )
    if( rst == 1'b1 )
        s_cur <= S_IDLE;
    else
        s_cur <= s_nxt;
always @ ( * ) begin
    s_nxt = S_IDLE
    case ( s_cur )
        S_IDLE:         begin
            s_nxt = S_IDLE;
            if( param_ena )
                s_nxt = S_READ_X2;
        end
        S_READ_X2:  begin
            s_nxt = S_READ_X2;
            if( ~select_x )
                s_nxt = S_WAIT_X;
        end
        S_WAIT_X:   s_nxt = S_WAIT_X;
        S_READ_X:   begin

        end
        default: s_nxt = S_IDLE;
    endcase
end


altsyncram	WRAM (
			.address_a (wraddress),
			.address_b (rdaddress),
			.clock0 (clock),
			.data_a (data),
			.wren_a (wren),
			.q_b (sub_wire0),
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
	altsyncram_component.address_aclr_b = "NONE",
	altsyncram_component.address_reg_b = "CLOCK0",
	altsyncram_component.clock_enable_input_a = "BYPASS",
	altsyncram_component.clock_enable_input_b = "BYPASS",
	altsyncram_component.clock_enable_output_b = "BYPASS",
	altsyncram_component.intended_device_family = "Cyclone V",
	altsyncram_component.lpm_type = "altsyncram",
	altsyncram_component.numwords_a = 256,
	altsyncram_component.numwords_b = 256,
	altsyncram_component.operation_mode = "DUAL_PORT",
	altsyncram_component.outdata_aclr_b = "NONE",
	altsyncram_component.outdata_reg_b = "UNREGISTERED",
	altsyncram_component.power_up_uninitialized = "FALSE",
	altsyncram_component.read_during_write_mode_mixed_ports = "OLD_DATA",
	altsyncram_component.widthad_a = 8,
	altsyncram_component.widthad_b = 8,
	altsyncram_component.width_a = 32,
	altsyncram_component.width_b = 32,
	altsyncram_component.width_byteena_a = 1;

endmodule

