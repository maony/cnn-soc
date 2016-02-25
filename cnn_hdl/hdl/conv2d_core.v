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
    parameter D_MUL5    = 5,
    parameter D_MUL11   = 11,
    parameter D_ADD     = 7
)(
    input                   param_ena,
    input    [KS*KS*32-1:0] param_weight,
    input     [C_WIDTH-1:0] param_width_in,
    input     [C_WIDTH-1:0] param_width_out,
    input    [C_LENGTH-1:0] param_length,

    input                   pxl_ena_x,
    input            [31:0] pxl_x,
    output                  pxl_ena_y,
    input            [31:0] pxl_y,
    output                  pxl_ena_z,
    output           [31:0] pxl_z,

    input                   clk,
    input                   rst
);

localparam C_COUNT  = 15;

reg           [C_COUNT-1:0] dly_write[0:2];
reg           [C_COUNT-1:0] dly_read[0:2];

reg          [KS*KS*32-1:0] weight_reg;
reg           [C_WIDTH-1:0] width_in;
reg           [C_WIDTH-1:0] width_out;
reg          [C_LENGTH-1:0] length;

wire                 [31:0] weight[0:KS*KS-1];
wire                 [31:0] result_mul[0:KS*KS-1];
wire                 [31:0] result_add[0:KS*KS-1];
reg                  [31:0] dly_mul[0:KS*6-1];
wire                 [31:0] dataa_add[0:KS*KS-1];
wire                 [31:0] datab_add[0:KS*KS-1];

genvar                      g;

reg                         run;
reg           [C_COUNT-1:0] dly_cnt;

wire                  [2:0] wreq_row;
reg                   [2:0] wreq_run;
reg          [C_LENGTH-1:0] wreq_cnt[0:2];
wire                  [1:0] rreq_row;
reg                   [2:0] rreq_run;
reg          [C_LENGTH-1:0] rreq_cnt[0:1];

reg           [C_WIDTH-1:0] out_cnt;

always @ ( posedge clk )
    if( pxl_ena_x ) begin
        dly_write[0] <= D_MUL5 + D_ADD*3;
        dly_write[1] <= width_in + D_MUL5 + D_ADD*3;
        dly_write[2] <= (width_in << 1) + D_MUL5 + D_ADD*3 + D_ADD;
        dly_read[0]  <= width_in + D_ADD - 3;
        dly_read[1]  <= (width_in << 1) + D_ADD - 3;
        dly_read[2]  <= (width_in << 1) + D_MUL5 + D_ADD*3;
    end

always @ ( posedge clk or posedge rst )
    if( rst )
        run <= 1'b0;
    else if( pxl_ena_x )
        run <= 1'b1;
    else if( pxl_ena_z )
        run <= 1'b0;

always @ ( posedge clk or posedge rst )
    if( param_ena ) begin
        weight_reg  <= param_weight;
        width_in    <= param_width_in;
        width_out   <= param_width_out;
        length      <= param_length;
    end

generate
    for( g = 0; g < KS*KS; g = g + 1) begin:WADD
        assign weight[g] = weight_reg[(g+1)*32-1:g*32];

        fp_add7 FP_ADD      (
            .clock          ( clk ),
            .dataa          ( dataa_add[g] ),
            .datab          ( datab_add[g] ),
            .result         ( result_add[g] )
        );
    end

    for( g = 0; g < KS; g = g + 1 ) begin:MUL
        fp_mul5 FP_MUL0     (
            .clock          ( clk ),
            .dataa          ( weight[KS*g+0] ),
            .datab          ( pxl_x ),
            .result         ( result_mul[KS*g+0] )
        );
        fp_mul11 FP_MUL1    (
            .clock          ( clk ),
            .dataa          ( weight[KS*g+1] ),
            .datab          ( pxl_x ),
            .result         ( result_mul[KS*g+1] )
        );
        fp_mul11 FP_MUL2    (
            .clock          ( clk ),
            .dataa          ( weight[KS*g+2] ),
            .datab          ( pxl_x ),
            .result         ( result_mul[KS*g+2] )
        );
    end
    
    for( g = 0; g < 5; g = g + 1) begin:DLY
        always @ ( posedge clk ) begin
            dly_mul[g+01] <= dly_mul[g+00];
            dly_mul[g+07] <= dly_mul[g+06];
            dly_mul[g+13] <= dly_mul[g+12];
        end
    end
endgenerate

always @ ( posedge clk ) begin
    dly_mul[0]  <= result_mul[2];
    dly_mul[6]  <= result_mul[5];
    dly_mul[12] <= result_mul[8];
end

assign dataa_add[0] = 32'd0;
assign dataa_add[1] = result_add[0];
assign dataa_add[2] = result_add[1];
assign dataa_add[4] = result_add[3];
assign dataa_add[5] = result_add[4];
assign dataa_add[7] = result_add[6];
assign dataa_add[8] = result_add[7];
assign datab_add[0] = result_mul[0];
assign datab_add[1] = result_mul[1];
assign datab_add[2] = dly_mul[5];
assign datab_add[3] = result_mul[3];
assign datab_add[4] = result_mul[4];
assign datab_add[5] = dly_mul[11];
assign datab_add[6] = result_mul[6];
assign datab_add[7] = result_mul[7];
assign datab_add[8] = dly_mul[17];

always @ ( posedge clk )
    if( pxl_ena_x )
        wreq_cnt[0] <= {{(C_LENGTH-1){1'b0}}, 1'b1};
    else if( wreq_row[0] )
        wreq_cnt[0] <= wreq_cnt[0] + 'd1;
assign wreq_row[0] = wreq_run[0];
always @ ( posedge clk or posedge rst ) begin
    if( pxl_ena_x || rst )
        wreq_run[0] <= 1'b0;
    else if( wreq_cnt[0] == length )
        wreq_run[0] <= 1'b0;
    else if( dly_cnt == dly_write[0] )
        wreq_run[0] <= 1'b1;
end

scfifo	SCFF0 (
    .aclr           ( rst ),
    .clock          ( clk ),
    .data           ( result_add[2] ),
    .rdreq          ( rreq_row[0] ),
    .sclr           ( 1'b0 ),
    .wrreq          ( wreq_row[0] ),
    .almost_empty   ( ),
    .almost_full    ( ),
    .empty          ( ),
    .full           ( ),
    .q              ( dataa_add[3] ),
    .usedw          ( ),
    .eccstatus ());
defparam
	SCFF0.add_ram_output_register = "OFF",
	SCFF0.almost_empty_value = 8,
	SCFF0.almost_full_value = 250,
	SCFF0.intended_device_family = "Cyclone V",
	SCFF0.lpm_hint = "RAM_BLOCK_TYPE=M10K",
	SCFF0.lpm_numwords = 256,
	SCFF0.lpm_showahead = "OFF",
	SCFF0.lpm_type = "scfifo",
	SCFF0.lpm_width = 32,
	SCFF0.lpm_widthu = 8,
	SCFF0.overflow_checking = "ON",
	SCFF0.underflow_checking = "ON",
	SCFF0.use_eab = "ON";

always @ ( posedge clk or posedge rst ) begin
    if( pxl_ena_x || rst )
        rreq_run[0] <= 1'b0;
    else if( rreq_cnt[0] == length )
        rreq_run[0] <= 1'b0;
    else if( dly_cnt == dly_read[0] )
        rreq_run[0] <= 1'b1;
end
always @ ( posedge clk )
    if( pxl_ena_x )
        rreq_cnt[0] <= {{(C_LENGTH-1){1'b0}}, 1'b1};
    else if( rreq_row[0] )
        rreq_cnt[0] <= rreq_cnt[0] + 'd1;
assign rreq_row[0] = rreq_run[0];

always @ ( posedge clk )
    if( pxl_ena_x )
        wreq_cnt[1] <= {{(C_LENGTH-1){1'b0}}, 1'b1};
    else if( wreq_row[1] )
        wreq_cnt[1] <= wreq_cnt[1] + 'd1;
assign wreq_row[1] = wreq_run[1];
always @ ( posedge clk or posedge rst ) begin
    if( pxl_ena_x || rst )
        wreq_run[1] <= 1'b0;
    else if( wreq_cnt[1] == length )
        wreq_run[1] <= 1'b0;
    else if( dly_cnt == dly_write[1] )
        wreq_run[1] <= 1'b1;
end

scfifo	SCFF1 (
    .aclr           ( rst ),
    .clock          ( clk ),
    .data           ( result_add[5] ),
    .rdreq          ( rreq_row[1] ),
    .sclr           ( 1'b0 ),
    .wrreq          ( wreq_row[1] ),
    .almost_empty   ( ),
    .almost_full    ( ),
    .empty          ( ),
    .full           ( ),
    .q              ( dataa_add[6] ),
    .usedw          ( ),
    .eccstatus ());
defparam
	SCFF1.add_ram_output_register = "OFF",
	SCFF1.almost_empty_value = 8,
	SCFF1.almost_full_value = 250,
	SCFF1.intended_device_family = "Cyclone V",
	SCFF1.lpm_hint = "RAM_BLOCK_TYPE=M10K",
	SCFF1.lpm_numwords = 256,
	SCFF1.lpm_showahead = "OFF",
	SCFF1.lpm_type = "scfifo",
	SCFF1.lpm_width = 32,
	SCFF1.lpm_widthu = 8,
	SCFF1.overflow_checking = "ON",
	SCFF1.underflow_checking = "ON",
	SCFF1.use_eab = "ON";

always @ ( posedge clk or posedge rst )
    if( pxl_ena_x || rst )
        rreq_run[1] <= 1'b0;
    else if( rreq_cnt[1] == length )
        rreq_run[1] <= 1'b0;
    else if( dly_cnt == dly_read[1] )
        rreq_run[1] <= 1'b1;
always @ ( posedge clk )
    if( pxl_ena_x )
        rreq_cnt[1] <= {{(C_LENGTH-1){1'b0}}, 1'b1};
    else if( rreq_row[1] )
        rreq_cnt[1] <= rreq_cnt[1] + 'd1;
assign rreq_row[1] = rreq_run[1];

always @ ( posedge clk )
    if( pxl_ena_x )
        wreq_cnt[2] <= {{(C_LENGTH-1){1'b0}}, 1'b1};
    else if( wreq_row[2] )
        wreq_cnt[2] <= wreq_cnt[2] + 'd1;
assign wreq_row[2] = wreq_run[2];
always @ ( posedge clk or posedge rst )
    if( pxl_ena_x || rst )
        wreq_run[2] <= 1'b0;
    else if( wreq_cnt[2] == length )
        wreq_run[2] <= 1'b0;
    else if( dly_cnt == dly_write[2] )
        wreq_run[2] <= 1'b1;
assign pxl_ena_z = wreq_row[2];

//always @ ( posedge clk or posedge rst )
//    if( rst )
//        rreq_run[2] <= 1'b0;
//    else if( dly_cnt == dly_read[2] )
//        rreq_run[2] <= 1'b1;
//    else
//        rreq_run[2] <= 1'b0;
always @ ( posedge clk or posedge rst )
    if( pxl_ena_x || rst )
        rreq_run[2] <= 1'b0;
    else if( rreq_cnt[2] == length )
        rreq_run[2] <= 1'b0;
    else if( dly_cnt == dly_read[2] )
        rreq_run[2] <= 1'b1;
always @ ( posedge clk )
    if( pxl_ena_x )
        rreq_cnt[2] <= {{(C_LENGTH-1){1'b0}}, 1'b1};
    else if( pxl_ena_y )
        rreq_cnt[2] <= rreq_cnt[2] + 'd1;
assign pxl_ena_y = rreq_run[2] && (!(out_cnt == width_out));   

// border process, cross row
always @ ( posedge clk )
    if( pxl_ena_x )
        out_flag <= 2'b0;
    else if( out_cnt == width_out )
        out_flag <= out_flag + 2'b1;
always @ ( posedge clk )
    if( pxl_ena_x || rst )
        out_cnt <= {{(C_WIDTH-1){1'b0}}, 1'b1};
    else if( out_flag == 2'b01 )
        out_cnt <= {{(C_WIDTH-1){1'b0}}, 1'b1};
    else if( pxl_ena_y )
        out_cnt <= out_cnt + 'd1;

// delay count
always @ ( posedge clk or rst )
    if( rst || pxl_ena_x )
        dly_cnt <= {{(C_COUNT-2){1'b0}}, 2'b10};
    else if( run )
        dly_cnt <= dly_cnt + 'd1;

fp_add7 FP_ADDZ     (
    .clock          ( clk ),
    .dataa          ( result_add[8] ),
    .datab          ( pxl_y ),
    .result         ( pxl_z )
);

endmodule
