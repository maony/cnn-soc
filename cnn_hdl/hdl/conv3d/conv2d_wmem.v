/*
* Created           : mny
* Date              : 201602
*/

// synposys translate_off
`timescale 1ns/100ps
// synposys translate_on

module conv2d_wmem #(
    parameter AW = 30,
    parameter DW = 256
)(
    output                          wmst_ctrl_fixed_location,
    output                 [AW-1:0] wmst_ctrl_write_base,
    output                 [AW-1:0] wmst_ctrl_write_length,
    output                          wmst_ctrl_go,
    input                           wmst_ctrl_done,

    output                          wmst_user_write_buffer,
    output                 [DW-1:0] wmst_user_write_input_data,
    input                           wmst_user_buffer_full,

    input               param_ena,
    input      [AW-1:0] param_zaddr,
    input        [17:0] param_length_out,

    input               pxl_ena_z,
    input        [31:0] pxl_z,
    output              flag_write_over,

    input           rst,
    input           clk
);

reg               [1:0] dly_ena;
reg              [15:0] cnt_unit;
reg              [15:0] cnt_len;
reg            [AW-1:0] write_base;
reg                     wrreq;
reg              [31:0] data;
reg               [2:0] cnt_word;
reg                     rdreq;
reg                     rdreq_dly;
reg                     write_buf_req;
reg            [DW-1:0] write_buf_dat;
reg               [4:0] write_len;
reg                     go;

always @ ( posedge clk ) begin
    dly_ena[0] <= param_ena;
    dly_ena[1] <= (~dly_ena[0]) & param_ena;
    if( dly_ena[1] ) 
        cnt_unit   <= param_length_out[17:2];
end
always @ ( posedge clk )
    if( dly_ena[1]) )
        write_base <= param_zaddr;
    else if( go )
        write_base <= write_base + {write_len, 2'b0};
always @ ( posedge clk ) begin
    wrreq   <= pxl_ena_z;
    data    <= pxl_z;
end

always @ ( posedge clk or posedge rst )
    if( rst == 1'b1 )
        cnt_word <= 3'd0;
    else if( (usedw > 'd3) && (cnt_word == 'd0) )
        cnt_word <= 'd4;
    else if( cnt_word > 'd0 )
        cnt_word <= cnt_word - 'd1;
always @ ( posedge clk or posedge rst ) begin
    rdreq_dly <= rdreq;
    if( rst == 1'b1 )
        rdreq <= 1'b0;
    else if( cnt_word > 'd0 )
        rdreq <= 1'b1;
    else
        rdreq <= 1'b0;
end
always @ ( posedge clk or posedge rst )
    if( rst == 1'b1 )
        write_buf_req <= 1'b0;
    else if( rdreq_dly & (~rdreq))
        write_buf_req <= 1'b1;
    else
        write_buf_req <= 1'b0;
always @ ( posedge clk or posedge rst )
    if( rst == 1'b1 )
        write_buf_dat <= 'd0;
    else if( rdreq_dly )
        write_buf_dat <= {q, write_buf_dat[127:32]};
always @ ( posedge clk or posedge rst )
    if( rst == 1'b1 )
        write_len <= 'd0;
    else if( write_buf_req ) begin
        if( go )
            write_len <= 'd1;
        else
            write_len <= write_len + 'd1;
    end
    else if( go )
        write_len <= 'd0;
always @ ( posedge clk or posedge rst )
    if( rst == 1'b1 )
        cnt_len <= 'd0;
    else if( dly_ena[1] )
        cnt_len <= 'd0;
    else if( write_buf_req )
        cnt_len <= cnt_len + 'd1;
always @ ( posedge clk or posedge rst )
    if( rst == 1'b1 )
        go <= 1'b0;
    else if( ((cnt_len == cnt_unit) || (write_len > 'd15)) && (wmst_ctrl_done) )
        go <= 1'b1;
    else 
        go <= 1'b0;

assign wmst_ctrl_fixed_location = 1'b0;
assign wmst_ctrl_write_base = write_base;
assign wmst_ctrl_write_length = write_len;
assign wmst_ctrl_go = go;
assign wmst_user_write_buffer = write_buf_req;
assign wmst_user_write_input_data = write_buf_dat;
assign flag_write_over = (cnt_len == cnt_unit) & go;

scfifo	SCFF (
    .aclr           ( rst ),
    .clock          ( clk ),
    .data           ( data ),
    .rdreq          ( rdreq ),
    .sclr           ( 1'b0 ),
    .wrreq          ( wrreq ),
    .almost_empty   ( ),
    .almost_full    ( ),
    .empty          ( ),
    .full           ( ),
    .q              ( q ),
    .usedw          ( usedw ),
    .eccstatus ());
defparam
    SCFF.add_ram_output_register = "OFF",
    SCFF.almost_empty_value = 240,
    SCFF.almost_full_value = 250,
    SCFF.intended_device_family = "Cyclone V",
    SCFF.lpm_hint = "RAM_BLOCK_TYPE=M10K",
    SCFF.lpm_numwords = 256,
    SCFF.lpm_showahead = "OFF",
    SCFF.lpm_type = "scfifo",
    SCFF.lpm_width = 32,
    SCFF.lpm_widthu = 8,
    SCFF.overflow_checking = "ON",
    SCFF.underflow_checking = "ON",
    SCFF.use_eab = "ON";

endmodule

