/*
* Created           : mny
* Date              : 201602
*/

// synposys translate_off
`timescale 1ns/100ps
// synposys translate_on

module conv2d_core #(
    parameter C_WIDTH   = 9,
    parameter C_LENGTH  = 2 * C_WIDTH,
    parameter KS        = 3,
    parameter D_MUL     = 6,
    parameter D_ADD     = 10
)(
    input                   param_ena,
    input    [KS*KS*32-1:0] param_weight,
    input     [C_WIDTH-1:0] param_width,
    input    [C_LENGTH-1:0] param_length,

    input                   pxl_ena_in,
    input            [31:0] pxl_x,
    input            [31:0] pxl_y,
    output                  pxl_ena_out,
    output           [31:0] pxl_z,

    input                   clk,
    input                   rst
);

localparam C_COUNT  = 5;
localparam D_R0_W   = (D_MUL+D_ADD) << 1 + (D_MUL + D_ADD) << 0 + 2;
localparam D_R1_W   = (D_MUL+D_ADD) << 2 + (D_MUL + D_ADD) << 1 + 4 - KS;
localparam D_R2_W   = (D_MUL+D_ADD) << 3 + (D_MUL + D_ADD) << 1 + 6 - KS << 1;

reg           [C_COUNT-1:0] dly_write[3];
reg           [C_COUNT-1:0] dly_read[2];

reg          [KS*KS*32-1:0] weight;
reg           [C_WIDTH-1:0] width;
reg          [C_LENGTH-1:0] length;

wire         [KS*KS*32-1:0] result_mul;
wire         [KS*KS*32-1:0] result_add;
reg          [KS*KS*32-1:0] result_dly;

genvar                      g;

reg                         run;
reg           [C_COUNT-1:0] dly_cnt;

wire                  [2:0] wreq_row;
reg                   [2:0] wreq_run;
reg          [C_LENGTH-1:0] wreq_cnt[3];
wire             [2*32-1:0] data_row;
wire                  [1:0] rreq_row;
reg                   [1:0] rreq_run;
reg          [C_LENGTH-1:0] rreq_cnt[2];

always @ ( posedge clk or posedge rst )
    if( rst )
        run <= 1'b0;
    else if( pxl_ena_in )
        run <= 1'b1;
    else if( pxl_ena_out )
        run <= 1'b0;

always @ ( posedge clk or posedge rst )
    if( param_ena ) begin
        weight <= param_weight;
        width  <= param_width;
        length <= param_length;
    end

always @ ( posedge clk )
    if( pxl_ena_in ) begin
        dly_write[0] <= D_R0_W;
        dly_write[1] <= D_R1_W + width;
        dly_write[2] <= D_R2_W + width << 1 + D_ADD;
        dly_read[0]  <= D_R0_W + width - KS;
        dly_read[1]  <= D_R1_W + width << 1 - KS;
    end

generate
    for( g = 0; g < KS*KS; g = g + 1 )
    begin:MADD
        fp_mult FP_MULT     (
            .clock          ( clk ),
            .dataa          ( weight[(g+1)*32-1:g*32] ),
            .datab          ( pxl_x ),
            .result         ( result_mul[(g+1)*32-1:g*32] )
        );
        
        fp_add FP_ADD       (
            .clock          ( clk ),
            .dataa          ( result_dly[(g+1)*32-1:g*32] ),
            .datab          ( result_mul[(g+1)*32-1:g*32] ),
            .result         ( result_add[(g+1)*32-1:g*32] )
        );
    end
endgenerate

always @ ( posedge clk ) begin
    result_dly[1*32-1:0*32] <= 32'd0;
    result_dly[2*32-1:1*32] <= result_add[1*32-1:0*32];
    result_dly[3*32-1:2*32] <= result_add[2*32-1:1*32];
end

always @ ( posedge clk )
    if( pxl_ena_in )
        wreq_cnt[0] <= {{(C_LENGTH-1){1'b0}}, 1'b1};
    else if( wreq_row[0] )
        wreq_cnt[0] <= wreq_cnt[0] + 'd1;
assign wreq_row[0] = wreq_run[0];
always @ ( posedge clk or posedge rst ) begin
    if( pxl_ena_in || rst )
        wreq_run[0] <= 1'b0;
    else if( wreq_cnt[0] == length )
        wreq_run[0] <= 1'b0;
    else if( dly_cnt == dly_write[0] )
        wreq_run[0] <= 1'b1;
end

scfifo	SCFF0 (
    .aclr           ( rst ),
    .clock          ( clk ),
    .data           ( result_add[3*32-1:2*32] ),
    .rdreq          ( rreq_row[0] ),
    .sclr           ( 1'b0 ),
    .wrreq          ( wreq_row[0] ),
    .almost_empty   ( ),
    .almost_full    ( ),
    .empty          ( ),
    .full           ( ),
    .q              ( data_row[31:0] ),
    .usedw          ( ),
    .eccstatus ());
defparam
	scfifo_component.add_ram_output_register = "OFF",
	scfifo_component.almost_empty_value = 8,
	scfifo_component.almost_full_value = 250,
	scfifo_component.intended_device_family = "Cyclone V",
	scfifo_component.lpm_hint = "RAM_BLOCK_TYPE=M10K",
	scfifo_component.lpm_numwords = 256,
	scfifo_component.lpm_showahead = "OFF",
	scfifo_component.lpm_type = "scfifo",
	scfifo_component.lpm_width = 32,
	scfifo_component.lpm_widthu = 8,
	scfifo_component.overflow_checking = "ON",
	scfifo_component.underflow_checking = "ON",
	scfifo_component.use_eab = "ON";

always @ ( posedge clk or posedge rst ) begin
    if( pxl_ena_in || rst )
        rreq_run[0] <= 1'b0;
    else if( rreq_cnt[0] == length )
        rreq_run[0] <= 1'b0;
    else if( dly_cnt == dly_read[0] )
        rreq_run[0] <= 1'b1;
end
always @ ( posedge clk )
    if( pxl_ena_in )
        rreq_cnt[0] <= {{(C_LENGTH-1){1'b0}}, 1'b1};
    else if( rreq_row[0] )
        rreq_cnt[0] <= rreq_cnt[0] + 'd1;
assign rreq_row[0] = rreq_run[0];

always @ ( posedge clk ) begin
    result_dly[4*32-1:3*32] <= data_row[31:0];
    result_dly[5*32-1:4*32] <= result_add[4*32-1:3*32];
    result_dly[6*32-1:5*32] <= result_add[5*32-1:4*32];
end

always @ ( posedge clk )
    if( pxl_ena_in )
        wreq_cnt[1] <= {{(C_LENGTH-1){1'b0}}, 1'b1};
    else if( wreq_row[1] )
        wreq_cnt[1] <= wreq_cnt[1] + 'd1;
assign wreq_row[1] = wreq_run[1];
always @ ( posedge clk or posedge rst ) begin
    if( pxl_ena_in || rst )
        wreq_run[1] <= 1'b0;
    else if( wreq_cnt[1] == length )
        wreq_run[1] <= 1'b0;
    else if( dly_cnt == dly_write[1] )
        wreq_run[1] <= 1'b1;
end

scfifo	SCFF1 (
    .aclr           ( rst ),
    .clock          ( clk ),
    .data           ( result_add[6*32-1:5*32] ),
    .rdreq          ( rreq_row[1] ),
    .sclr           ( 1'b0 ),
    .wrreq          ( wreq_row[1] ),
    .almost_empty   ( ),
    .almost_full    ( ),
    .empty          ( ),
    .full           ( ),
    .q              ( data_row[2*32-1:1*32] ),
    .usedw          ( ),
    .eccstatus ());
defparam
	scfifo_component.add_ram_output_register = "OFF",
	scfifo_component.almost_empty_value = 8,
	scfifo_component.almost_full_value = 250,
	scfifo_component.intended_device_family = "Cyclone V",
	scfifo_component.lpm_hint = "RAM_BLOCK_TYPE=M10K",
	scfifo_component.lpm_numwords = 256,
	scfifo_component.lpm_showahead = "OFF",
	scfifo_component.lpm_type = "scfifo",
	scfifo_component.lpm_width = 32,
	scfifo_component.lpm_widthu = 8,
	scfifo_component.overflow_checking = "ON",
	scfifo_component.underflow_checking = "ON",
	scfifo_component.use_eab = "ON";

always @ ( posedge clk or posedge rst ) begin
    if( pxl_ena_in || rst )
        rreq_run[1] <= 1'b0;
    else if( rreq_cnt[1] == length )
        rreq_run[1] <= 1'b0;
    else if( dly_cnt == dly_read[1] )
        rreq_run[1] <= 1'b1;
end
always @ ( posedge clk )
    if( pxl_ena_in )
        rreq_cnt[1] <= {{(C_LENGTH-1){1'b0}}, 1'b1};
    else if( rreq_row[1] )
        rreq_cnt[1] <= rreq_cnt[1] + 'd1;
assign rreq_row[1] = rreq_run[1];

always @ ( posedge clk ) begin
    result_dly[7*32-1:6*32] <= data_row[63:32];
    result_dly[8*32-1:7*32] <= result_add[7*32-1:6*32];
    result_dly[9*32-1:8*32] <= result_add[8*32-1:7*32];
end

always @ ( posedge clk )
    if( pxl_ena_in )
        wreq_cnt[2] <= {{(C_LENGTH-1){1'b0}}, 1'b1};
    else if( wreq_row[2] )
        wreq_cnt[2] <= wreq_cnt[2] + 'd1;
assign wreq_row[2] = wreq_run[2];
always @ ( posedge clk or posedge rst )
    if( pxl_ena_in || rst )
        wreq_run[2] <= 1'b0;
    else if( wreq_cnt[2] == length )
        wreq_run[2] <= 1'b0;
    else if( dly_cnt == dly_write[2] )
        wreq_run[2] <= 1'b1;
assign pxl_ena_out = wreq_row[2];

always @ ( posedge clk or rst )
    if( rst || pxl_ena_in )
        dly_cnt <= {C_COUNT{1'b1}};
    else if( run )
        dly_cnt <= dly_cnt + 'd1;

fp_add FP_ADDZ      (
    .clock          ( clk ),
    .dataa          ( result_add[KS*KS*32-1:(KS*KS-1)*32] ),
    .datab          ( pxl_y ),
    .result         ( pxl_z )
);

endmodule
