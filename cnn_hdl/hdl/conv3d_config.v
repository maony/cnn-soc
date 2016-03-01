/*
* Created           : mny
* Date              : 201602
*/

// synposys translate_off
`timescale 1ns/100ps
// synposys translate_on

module conv3d_config #(
    parameter AW = 30,
    parameter DW = 256
)(
    input               config_ena,
    input         [5:0] config_addr,
    input        [31:0] config_data,
    
    output              cfg_ena,
    output reg [AW-1:0] cfg_xbase,
    output reg [AW-1:0] cfg_ybase,
    output reg [AW-1:0] cfg_zbase,
    output reg [AW-1:0] cfg_xoffset,
    output reg [AW-1:0] cfg_yoffset,
    output reg    [8:0] cfg_width_in,
    output reg    [8:0] cfg_height_out,
    output reg   [17:0] cfg_length_in,
    output reg   [17:0] cfg_length_out,
    
    output              cfg_prefetch,
    output reg [AW-1:0] cfg_waddr,
    output reg    [7:0] cfg_length_w,

    input           rst,
    input           clk
);

localparam WADDR        = 'd0;
localparam WLENGTH      = 'd1;
localparam WPREFETCH    = 'd2;

localparam XBASE        = 'd10;
localparam YBASE        = 'd11;
localparam ZBASE        = 'd12;
localparam XOFFSET      = 'd13;
localparam YOFFSET      = 'd14;
localparam WIDTH_IN     = 'd15;
localparam WIDTH_OUT    = 'd16;
localparam LENGTH_IN    = 'd17;
localparam LENGTH_OUT   = 'd18;
localparam CONVRUN      = 'd19;

reg             [1:0] prefetch;
reg             [1:0] conv_run; 

always @ ( posedge clk )
    if( config_ena && (config_addr == WADDR) )
        cfg_waddr <= config_data[AW-1:0];
always @ ( posedge clk )
    if( config_ena && (config_addr == WLENGTH) )
        cfg_length_w <= config_data[7:0];
always @ ( posedge clk )    begin
    prefetch[1] <= prefetch[0];
    if( config_ena && (config_addr == WPREFETCH) )
        prefetch[0] <= config_data[0];
end
assign cfg_prefetch = (~prefetch[1]) & prefetch[0];

always @ ( posedge clk )
    if( config_ena && (config_addr == XBASE) )
        cfg_xbase <= config_data[AW-1:0];
always @ ( posedge clk )
    if( config_ena && (config_addr == YBASE) )
        cfg_ybase <= config_data[AW-1:0];
always @ ( posedge clk )
    if( config_ena && (config_addr == ZBASE) )
        cfg_zbase <= config_data[AW-1:0];
always @ ( posedge clk )
    if( config_ena && (config_addr == XOFFSET) )
        cfg_xoffset <= config_data[AW-1:0];
always @ ( posedge clk )
    if( config_ena && (config_addr == YOFFSET) )
        cfg_yoffset <= config_data[AW-1:0];
always @ ( posedge clk )
    if( config_ena && (config_addr == WIDTH_IN) )
        cfg_width_in <= config_data[AW-1:0];
always @ ( posedge clk )
    if( config_ena && (config_addr == WIDTH_OUT) )
        cfg_width_out <= config_data[AW-1:0];
always @ ( posedge clk )
    if( config_ena && (config_addr == LENGTH_IN) )
        cfg_length_in <= config_data[AW-1:0];
always @ ( posedge clk )
    if( config_ena && (config_addr == LENGTH_OUT) )
        cfg_length_out <= config_data[AW-1:0];
always @ ( posedge clk )    begin
    conv_run[1] <= conv_run[0];
    if( config_ena && (config_addr == CONVRUN) )
        conv_run[0] <= config_data[0];
end
assign cfg_ena = (~conv_run[1]) & conv_run[0];
endmodule

