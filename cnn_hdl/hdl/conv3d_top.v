/*
* Created           : mny
* Date              : 201602
*/

// synposys translate_off
`timescale 1ns/100ps
// synposys translate_on

module conv3d_top #(
    parameter AW = 30,
    parameter DW = 128,
    parameter KS = 3
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
    
    input               cfg_prefetch,
    input      [AW-1:0] cfg_waddr,
    input         [7:0] cfg_length_w,
    
    output                          rmst_ctrl_fixed_location,
    output                 [AW-1:0] rmst_ctrl_read_base,
    output                 [AW-1:0] rmst_ctrl_read_length,
    output                          rmst_ctrl_go,
    input                           rmst_ctrl_done,
    output                          rmst_user_read_buffer,
    input                  [DW-1:0] rmst_user_buffer_data,
    input                           rmst_user_data_available,
    
    output                          wmst_ctrl_fixed_location,
    output                 [AW-1:0] wmst_ctrl_write_base,
    output                 [AW-1:0] wmst_ctrl_write_length,
    output                          wmst_ctrl_go,
    input                           wmst_ctrl_done,
    output                          wmst_user_write_buffer,
    output                 [DW-1:0] wmst_user_write_input_data,
    input                           wmst_user_buffer_full,

    input               clk,
    input               rst
);

wire              param_ena;
wire     [AW-1:0] param_xaddr;
wire     [AW-1:0] param_xaddr;
wire     [AW-1:0] param_zaddr;
wire        [8:0] param_width_in;
wire        [8:0] param_height_out;
wire       [17:0] param_length_in;
wire       [17:0] param_length_out;
wire              flag_write_over;  
wire   [KS*KS*32-1:0] pxl_w;
wire                  pxl_ena_x;
wire           [31:0] pxl_x;
wire                  pxl_ena_y;
wire           [31:0] pxl_y;

conv3d_schedule SCHEDULE(
    .cfg_ena            ( cfg_ena           ),
    .cfg_xbase          ( cfg_xbase         ),
    .cfg_ybase          ( cfg_ybase         ),
    .cfg_zbase          ( cfg_zbase         ),
    .cfg_xoffset        ( cfg_xoffset       ),
    .cfg_yoffset        ( cfg_yoffset       ),
    .cfg_width_in       ( cfg_width_in      ),
    .cfg_height_out     ( cfg_height_out    ),
    .cfg_length_in      ( cfg_length_in     ),
    .cfg_length_out     ( cfg_length_out    ),

    .param_ena          ( param_ena        ),
    .param_xaddr        ( param_xaddr      ),
    .param_xaddr        ( param_xaddr      ),
    .param_zaddr        ( param_zaddr      ),
    .param_width_in     ( param_width_in   ),
    .param_height_out   ( param_height_out ),
    .param_length_in    ( param_length_in  ),
    .param_length_out   ( param_length_out ),
    .flag_write_over    ( flag_write_over  ),  

    .clk( clk ),
    .rst( rst )
);

conv2d_rmem RMEM(
    .rmst_ctrl_fixed_location   ( rmst_ctrl_fixed_location ),
    .rmst_ctrl_read_base        ( rmst_ctrl_read_base      ),
    .rmst_ctrl_read_length      ( rmst_ctrl_read_length    ),
    .rmst_ctrl_go               ( rmst_ctrl_go             ),
    .rmst_ctrl_done             ( rmst_ctrl_done           ),

    .rmst_user_read_buffer      ( rmst_user_read_buffer    ),
    .rmst_user_buffer_data      ( rmst_user_buffer_data    ),
    .rmst_user_data_available   ( rmst_user_data_available ),
    
    .param_prefetch  ( cfg_prefetch ),
    .param_waddr     ( cfg_waddr    ),
    .param_length_w  ( cfg_length_w ),

    .param_ena          ( param_ena        ),
    .param_xaddr        ( param_xaddr      ),
    .param_yaddr        ( param_yaddr      ),
    .param_width_in     ( param_width_in   ),
    .param_length_in    ( param_length_in  ),
    .param_length_out   ( param_length_out ),

    .pxl_w          ( pxl_w     ),
    .pxl_ena_x      ( pxl_ena_x ),
    .pxl_x          ( pxl_x     ),
    .pxl_ena_y      ( pxl_ena_y ),
    .pxl_y          ( pxl_y     ),

    .rst( clk ),
    .clk( rst )
);

conv2d_core CORE(
    .param_ena         ( param_ena        ),
    .param_width_in    ( param_width_in   ),
    .param_height_out  ( param_height_out ),
    
    .param_weight       ( pxl_w     ),
    .pxl_ena_x          ( pxl_ena_x ),
    .pxl_x              ( pxl_x     ),
    .pxl_ena_y          ( pxl_ena_y ),
    .pxl_y              ( pxl_y     ),
    .pxl_ena_z          ( pxl_ena_z ),
    .pxl_z              ( pxl_z     ),
    .pxl_ovr            ( ),

    .clk ( clk ),
    .rst ( rst )
);

conv2d_wmem WMEM(
    .wmst_ctrl_fixed_location   ( wmst_ctrl_fixed_location ),
    .wmst_ctrl_write_base       ( wmst_ctrl_write_base     ),
    .wmst_ctrl_write_length     ( wmst_ctrl_write_length   ),
    .wmst_ctrl_go               ( wmst_ctrl_go             ),
    .wmst_ctrl_done             ( wmst_ctrl_done           ),

    .wmst_user_write_buffer     ( wmst_user_write_buffer     ),
    .wmst_user_write_input_data ( wmst_user_write_input_data ),
    .wmst_user_buffer_full      ( wmst_user_buffer_full      ),

    .param_ena          ( param_ena        ),
    .param_zaddr        ( param_zaddr      ),
    .param_length_out   ( param_length_out ),

    .pxl_ena_z          ( pxl_ena_z       ),
    .pxl_z              ( pxl_z           ),
    .flag_write_over    ( flag_write_over ),

    .rst( rst ),
    .clk( clk )
);

endmodule

