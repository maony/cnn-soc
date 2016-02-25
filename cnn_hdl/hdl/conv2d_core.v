/*
* Created           : mny
* Date              : 201602
*/

// synposys translate_off
`timescale 1ns/100ps
// synposys translate_on

module conv2d_core #(
    parameter C_WIDTH   = 9,
    parameter KS        = 3,
    parameter D_MUL5    = 5,
    parameter D_MUL11   = 11,
    parameter D_ADD     = 7
)(
    input                   param_ena,
    input    [KS*KS*32-1:0] param_weight,
    input     [C_WIDTH-1:0] param_width_in,

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
localparam D_PPL    = D_MUL5 + D_ADD * 3 + 1 * 2;

reg         [D_PPL+D_ADD:0] dly_ena_x[0:KS-1];
wire                  [0:0] run;
reg                [KS-1:0] dly_pxl_ena;
reg           [C_COUNT-1:0] dly_cnt;
reg                  [31:0] pxl_x_reg;

reg          [KS*KS*32-1:0] weight_reg;
reg           [C_WIDTH-1:0] width_in;

wire                 [31:0] weight[0:KS*KS-1];
wire                 [31:0] result_mul[0:KS*KS-1];
wire                 [31:0] result_add[0:KS*KS-1];
reg                  [31:0] result_add_dly[0:KS*(KS-1)];
reg                  [31:0] dly_mul0[0:2];
reg                  [31:0] dly_mul1[0:KS*8-1];
wire                 [31:0] dataa_add[0:KS*KS-1];
wire                 [31:0] datab_add[0:KS*KS-1];

genvar                      g;

always @ ( posedge clk )
    dly_pxl_ena[0] <= pxl_ena_x;
always @ ( posedge clk or rst)
    if( param_ena || rst )
        dly_pxl_ena[1] <= 1'b0;
    else if(dly_cnt == ({6'd0, width_in}))
        dly_pxl_ena[1] <= 1'b1;
always @ ( posedge clk or rst)
    if( param_ena || rst )
        dly_pxl_ena[2] <= 1'b0;
    else if(dly_cnt == ({5'd0, width_in, 1'b0}))
        dly_pxl_ena[2] <= 1'b1;

assign run[0] = (~dly_pxl_ena[0]) && pxl_ena_x;

// delay count
always @ ( posedge clk or rst )
    if( rst || run[0] )
        dly_cnt <= {{(C_COUNT-2){1'b0}}, 2'b10};
    else if( pxl_ena_x )
        dly_cnt <= dly_cnt + 'd1;

always @ ( posedge clk or posedge rst )
    if( param_ena ) begin
        weight_reg  <= param_weight;
        width_in    <= param_width_in;
    end

always @ ( posedge clk )
    pxl_x_reg <= pxl_x;
    
generate
    always @ ( posedge clk ) begin
        dly_ena_x[0][0] <= pxl_ena_x;
        dly_ena_x[1][0] <= pxl_ena_x & dly_pxl_ena[1];
        dly_ena_x[2][0] <= pxl_ena_x & dly_pxl_ena[2];
    end
    for( g = 0; g < D_PPL+D_ADD; g = g + 1) begin:MPPL
        always @ ( posedge clk ) begin
            dly_ena_x[0][g+1] <= dly_ena_x[0][g];
            dly_ena_x[1][g+1] <= dly_ena_x[1][g];
            dly_ena_x[2][g+1] <= dly_ena_x[2][g];
        end
    end

    for( g = 0; g < KS*KS; g = g + 1) begin:MADD
        assign weight[g] = weight_reg[(g+1)*32-1:g*32];

        fp_add7 FP_ADD      (
            .clock          ( clk ),
            .dataa          ( dataa_add[g] ),
            .datab          ( datab_add[g] ),
            .result         ( result_add[g] )
        );
    end
    for( g = 0; g < KS; g = g + 1) begin:MADDD
        always @ ( posedge clk ) 
            if( dly_ena_x[g][D_MUL5+D_ADD-1] )
                result_add_dly[g*(KS-1)]    <= result_add[g*KS];
        always @ ( posedge clk ) 
            if( dly_ena_x[g][D_MUL5+D_ADD*2-1] )
                result_add_dly[g*(KS-1)+1]  <= result_add[g*KS+1];
    end

    for( g = 0; g < KS; g = g + 1 ) begin:MMUL
        fp_mul5 FP_MUL0     (
            .clock          ( clk ),
            .dataa          ( weight[KS*g+0] ),
            .datab          ( pxl_x_reg ),
            .result         ( result_mul[KS*g+0] )
        );
        fp_mul11 FP_MUL1    (
            .clock          ( clk ),
            .dataa          ( weight[KS*g+1] ),
            .datab          ( pxl_x_reg ),
            .result         ( result_mul[KS*g+1] )
        );
        fp_mul11 FP_MUL2    (
            .clock          ( clk ),
            .dataa          ( weight[KS*g+2] ),
            .datab          ( pxl_x_reg ),
            .result         ( result_mul[KS*g+2] )
        );
    end
    
    for( g = 0; g < 7; g = g + 1) begin:MDLY
        always @ ( posedge clk ) begin
            dly_mul1[g+01] <= dly_mul1[g+00];
            dly_mul1[g+09] <= dly_mul1[g+08];
            dly_mul1[g+17] <= dly_mul1[g+16];
        end
    end
    
    for( g = 0; g < KS-1; g = g + 1) begin:MFF
        scfifo	SCFF (
            .aclr           ( rst ),
            .clock          ( clk ),
            .data           ( result_add[g*KS+2] ),
            .rdreq          ( dly_ena_x[g+1][4] ),
            .sclr           ( 1'b0 ),
            .wrreq          ( dly_ena_x[g][D_PPL] ),
            .almost_empty   ( ),
            .almost_full    ( ),
            .empty          ( ),
            .full           ( ),
            .q              ( dataa_add[(g+1)*KS] ),
            .usedw          ( ),
            .eccstatus ());
        defparam
	        SCFF.add_ram_output_register = "OFF",
	        SCFF.almost_empty_value = 8,
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
    end

    for( g = 0; g < KS; g = g+1) begin: MASSA
        assign dataa_add[g*KS+1] = result_add_dly[2*g];
        assign dataa_add[g*KS+2] = result_add_dly[2*g+1];
        assign datab_add[g*KS+0] = result_mul[g*KS];
        assign datab_add[g*KS+1] = dly_mul0[g];
        assign datab_add[g*KS+2] = dly_mul1[(g+1)*(D_ADD+1)-1];
    end
endgenerate

always @ ( posedge clk ) begin
    dly_mul0[0]  <= result_mul[1];
    dly_mul0[1]  <= result_mul[4];
    dly_mul0[2]  <= result_mul[7];

    dly_mul1[0]  <= result_mul[2];
    dly_mul1[8]  <= result_mul[5];
    dly_mul1[16] <= result_mul[8];
end

assign dataa_add[0] = 32'd0;

assign pxl_ena_y = dly_ena_x[2][D_PPL-1];
assign pxl_ena_z = dly_ena_x[2][D_PPL+D_ADD];

fp_add7 FP_ADDZ     (
    .clock          ( clk ),
    .dataa          ( result_add[8] ),
    .datab          ( pxl_y ),
    .result         ( pxl_z )
);

endmodule
