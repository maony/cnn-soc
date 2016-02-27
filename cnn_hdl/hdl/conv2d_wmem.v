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

reg               [0:0] dly_ena;
reg              [17:0] length_out;
reg            [AW-1:0] write_base;
reg            [DW-1:0] write_data;
reg               [1:0] cnt_write;
reg                     write_flag;
reg                     write_ena;
reg               [0:0] done;
reg                     can_write;
reg            [AW-1:0] write_len;
reg              [17:0] cnt_len;
reg               [7:0] cnt_last;
reg                     go;
reg                     over;
reg               [3:0] dly_ovr;

always @ ( posedge clk ) begin
    dly_ena[0] <= param_ena;
    dly_ena[0] <= (~dly_ena[0]) & param_ena;
    if( (~dly_ena[0]) & param_ena 
        length_out <= param_length_out;
end
always @ ( posedge clk )
    if( (~dly_ena[0]) & param_ena )
        write_base <= param_zaddr;
    else if( go )
        write_base <= write_base + 'd16;
always @ ( posedge clk ) begin
    if( pxl_ena_z ) begin
        write_data <= write_data;
        case ( cnt_write )
            2'b00: write_data[31:0]     <= pxl_z;
            2'b01: write_data[63:32]    <= pxl_z;
            2'b10: write_data[95:64]    <= pxl_z;
            2'b11: write_data[127:96]   <= pxl_z;
    end
end
always @ ( posedge clk or posedge rst )
    if( rst == 1'b1 )
        cnt_write <= 2'b0;
    else if( pxl_ena_z )
        cnt_write <= cnt_write + 2'b1;
always @ ( posedge clk or posedge rst ) 
    if( ((~dly_ena[0]) & param_ena) || (rst == 1'b1) )
        write_flag <= 1'b0;
    else if( cnt_write[0] )
        write_flag <= 1'b1;
always @ ( posedge clk or posedge rst )
    if( rst == 1'b1 )
        write_ena <= 1'b0;
    else if( dly_ovr[1] )
        write_ena <= 1'b1;
    else if( (cnt_write == 2'b0) & pxl_ena_z & write_flag )
        write_ena <= 1'b1;
    else
        write_ena <= 1'b0;
always @ ( posedge clk or posedge rst )
    if( rst == 1'b1)
        cnt_last <= 'd0;
    else if( go ) begin
        if( write_ena )
            cnt_last <= cnt_last - 'd3;
        else
            cnt_last <= cnt_last + 'd1;
    end
    else if( write_ena )
        cnt_last <= cnt_last + 'd1;
always @ ( posedge clk )
    if( cnt_last > 'd3 )
        write_len <= 'd4;
    else
        write_len <= cnt_last;
always @ ( posedge clk or posedge rst ) begin
    done[0] <= wmst_ctrl_done;
    if( ((~dly_ena[0]) & param_ena) || (rst == 1'b1) )
        can_write <= 1'b1;
    else if( go )
        can_write <= 1'b0;
    else if( (~done[0]) & wmst_ctrl_done )
        can_write <= 1'b1;
end
always @ ( posedge clk or posedge rst ) begin
    dly_ovr[0] <= (cnt_len == length_out);
    dly_ovr[1] <= (~dly_ovr[0]) & (cnt_len == length_out);
    dly_ovr[2] <= dly_ovr[1];
    dly_ovr[3] <= dly_ovr[2];
    if( ((~dly_ena[0]) & param_ena) || (rst == 1'b1) )
        cnt_len <= 'd0;
    else if( pxl_ena_z )
        cnt_len <= cnt_len + 'd1;
end
always @ ( posedge clk )
    go <= ((cnt_last > 'd3) | dly_ovr[3]) & can_write;
always @ ( posedge clk )
    over <= (cnt_len == length_out) & can_write;

assign wmst_ctrl_fixed_location = 1'b0;
assign wmst_ctrl_write_base = write_base;
assign wmst_ctrl_write_length = write_len;
assign wmst_ctrl_go = go;
assign wmst_user_write_buffer = write_ena;
assign wmst_user_write_input_data = write_data;
assign flag_write_over = over;

endmodule

