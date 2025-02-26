module ram128x1 (clk, wea, aa, da, lda, qa, ab, ldb, qb);
    input clk;
    input wea;
    input [6:0] aa;
    input [0:0] da;
    input lda;
    input [6:0] ab;
    input ldb;
    output [0:0] qa;
    output [0:0] qb;

    wire [0:0] ao;
    wire [0:0] bo;
    reg [0:0] qa;
    reg [0:0] qb;

    RAM128X1D ram128x1_0 (
        .DPRA(ab),
        .A(aa),
        .DPRA(ab),
        .WE(wea),
        .WCLK(clk),
        .D(da[0]),
        .DPO(qb[0]),
        .SPO(ao[0])
    );

    always @(posedge clk) begin
        if (lda) qa <= ao;
        if (ldb) qb <= bo;
    end
endmodule