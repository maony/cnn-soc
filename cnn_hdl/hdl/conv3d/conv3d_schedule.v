/*
* Created           : mny
* Date              : 201602
*/

// synposys translate_off
`timescale 1ns/100ps
// synposys translate_on

module conv3d_schedule #(
    parameter AW = 128
)(
    input               cfg_ena,
    input      [AW-1:0] cfg_xbase,
    input      [AW-1:0] cfg_ybase,
    input      [AW-1:0] cfg_zbase,
    input      [AW-1:0] cfg_xoffset,
    input      [AW-1:0] cfg_yoffset,
    input         [8:0] cfg_width_in,
    input         [8:0] cfg_height_out,
    input        [17:0] cfg_length_in,
    input        [17:0] cfg_length_out,

    output              param_ena,
    output     [AW-1:0] param_xaddr,
    output     [AW-1:0] param_xaddr,
    output     [AW-1:0] param_zaddr,
    output        [8:0] param_width_in,
    output        [8:0] param_height_out,
    output       [17:0] param_length_in,
    output       [17:0] param_length_out,
    input               flag_write_over,  

    input               clk,
    input               rst
);

assign param_width_in   = width_in  ;
assign param_height_out = height_out;
assign param_length_in  = length_in ;
assign param_length_out = length_out;
assign param_xaddr      = xbase;
assign param_xaddr      = ybase;
assign param_zaddr      = zbase;
always @ ( posedge clk ) begin
    dly_ena[0] <= cfg_ena;
    dly_ena[1] <= (~dly_ena[0]) & cfg_ena;
    if( (~dly_ena[0]) & cfg_ena ) begin
        xoffset     <= cfg_xoffset;
        yoffset     <= cfg_yoffset;
        width_in    <= cfg_width_in;
        height_out  <= cfg_height_out;
        length_in   <= cfg_length_in;
        length_out  <= cfg_length_out;
    end
end
always @ ( posedge clk )
    if( (~dly_ena[0]) & cfg_ena )   begin
        xbase <= cfg_xbase;
        ybase <= cfg_ybase;
        zbase <= cfg_zbase;
    end
    else if( flag_write_over )      begin
        xbase <= xbase + xoffset;
        ybase <= ybase + yoffset;
        zbase <= zbase + yoffset;
    end
always @ ( posedge clk )
    ena <= dly_ena[1] | flag_write_over;

endmodule
