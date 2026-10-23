// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

`timescale 1ns / 1ps

// Controlled 2x2 complex gate over the whole register, one amplitude pair per clock.
//
// For every pair (i0, i1 = i0 | 1<<target) with all `ctrl` bits set in i0 the engine computes
//   out0 = m00 * a0 + m01 * a1,   out1 = m10 * a0 + m11 * a1
// with exact products and sums of 25-bit coefficients and 32-bit components, then
// round-half-up by 23 bits and saturates to 32 bits. This is the arithmetic of
// crates/t1000/src/qpu/{fixed,model}.rs.
//
// i0 and i1 differ in one bit, so they sit in opposite parity banks at (possibly different)
// addresses i>>1; every bank sees one read and one write per cycle. Pairs of one gate are
// disjoint, so the pipeline has no hazard. Pipeline (edges): 1 address, 2 bank read, 3 capture,
// 4 products, 5 sums, 6 round/saturate (write registers), 7 bank write.
module qpu_gate (
    input  wire        clk,
    input  wire        rst,
    input  wire        start,
    input  wire [3:0]  n,
    input  wire [3:0]  target,
    input  wire [13:0] ctrl,
    input  wire signed [24:0] m00r, m00i, m01r, m01i, m10r, m10i, m11r, m11i,
    output reg         busy,
    output reg  [12:0] ra0,
    output reg  [12:0] ra1,
    input  wire [63:0] rd0,
    input  wire [63:0] rd1,
    output reg         we0,
    output reg         we1,
    output reg  [12:0] wa0,
    output reg  [12:0] wa1,
    output reg  [63:0] wd0,
    output reg  [63:0] wd1
);

    // ---- pair generator -----------------------------------------------------
    reg  [12:0] p;
    reg         issue;
    wire [12:0] plast = (13'd1 << (n - 4'd1)) - 13'd1;

    wire [13:0] lowmask = (14'd1 << target) - 14'd1;
    wire [13:0] p14     = {1'b0, p};
    wire [13:0] i0      = ((p14 & ~lowmask) << 1) | (p14 & lowmask);
    wire [13:0] i1      = i0 | (14'd1 << target);
    wire        act     = ((i0 & ctrl) == ctrl);
    wire        par0    = ^i0;

    // meta = {act, par0, addr1[12:0], addr0[12:0]}
    reg [27:0] meta1, meta2, meta3, meta4, meta5;
    reg        v1 = 1'b0, v2 = 1'b0, v3 = 1'b0, v4 = 1'b0, v5 = 1'b0;

    // stage 1: bank addresses
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            ra0 <= 13'd0; ra1 <= 13'd0; meta1 <= 28'd0; v1 <= 1'b0;
        end else begin
            v1 <= issue;
            if (issue) begin
                ra0   <= par0 ? i1[13:1] : i0[13:1];
                ra1   <= par0 ? i0[13:1] : i1[13:1];
                meta1 <= {act, par0, i1[13:1], i0[13:1]};
            end
        end
    end

    // stage 2: bank read
    always @(posedge clk) begin
        meta2 <= meta1;
        v2    <= v1;
    end

    // stage 3: capture (a0 sits in bank par0)
    reg signed [31:0] a0r, a0i, a1r, a1i;
    always @(posedge clk) begin
        meta3 <= meta2;
        v3    <= v2;
        {a0i, a0r} <= meta2[26] ? rd1 : rd0;
        {a1i, a1r} <= meta2[26] ? rd0 : rd1;
    end

    // stage 4: products
    reg signed [56:0] q00rr, q00ii, q01rr, q01ii, q00ri, q00ir, q01ri, q01ir;
    reg signed [56:0] q10rr, q10ii, q11rr, q11ii, q10ri, q10ir, q11ri, q11ir;
    always @(posedge clk) begin
        meta4 <= meta3;
        v4    <= v3;
        q00rr <= m00r * a0r;  q00ii <= m00i * a0i;  q01rr <= m01r * a1r;  q01ii <= m01i * a1i;
        q00ri <= m00r * a0i;  q00ir <= m00i * a0r;  q01ri <= m01r * a1i;  q01ir <= m01i * a1r;
        q10rr <= m10r * a0r;  q10ii <= m10i * a0i;  q11rr <= m11r * a1r;  q11ii <= m11i * a1i;
        q10ri <= m10r * a0i;  q10ir <= m10i * a0r;  q11ri <= m11r * a1i;  q11ir <= m11i * a1r;
    end

    // stage 5: sums
    reg signed [59:0] s0r, s0i, s1r, s1i;
    always @(posedge clk) begin
        meta5 <= meta4;
        v5    <= v4;
        s0r <= q00rr - q00ii + q01rr - q01ii;
        s0i <= q00ri + q00ir + q01ri + q01ir;
        s1r <= q10rr - q10ii + q11rr - q11ii;
        s1i <= q10ri + q10ir + q11ri + q11ir;
    end

    // stage 6: round half up by 23 bits, saturate, steer to the banks
    function [31:0] rnd_sat(input signed [59:0] s);
        reg signed [59:0] t;
        begin
            t = (s + 60'sd4194304) >>> 23;
            if (t > 60'sd2147483647)        rnd_sat = 32'h7FFFFFFF;
            else if (t < -60'sd2147483648)  rnd_sat = 32'h80000000;
            else                            rnd_sat = t[31:0];
        end
    endfunction

    wire [63:0] o0 = {rnd_sat(s0i), rnd_sat(s0r)};
    wire [63:0] o1 = {rnd_sat(s1i), rnd_sat(s1r)};
    wire        mp = meta5[26];
    wire        ma = meta5[27];

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            we0 <= 1'b0; we1 <= 1'b0;
            wa0 <= 13'd0; wa1 <= 13'd0; wd0 <= 64'd0; wd1 <= 64'd0;
        end else begin
            we0 <= v5 & ma;
            we1 <= v5 & ma;
            wa0 <= mp ? meta5[25:13] : meta5[12:0];
            wa1 <= mp ? meta5[12:0]  : meta5[25:13];
            wd0 <= mp ? o1 : o0;
            wd1 <= mp ? o0 : o1;
        end
    end

    // ---- sequencing ---------------------------------------------------------
    reg [2:0] drain;
    wire      pipe_empty = ~(v1 | v2 | v3 | v4 | v5);

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            busy <= 1'b0; issue <= 1'b0; p <= 13'd0; drain <= 3'd0;
        end else begin
            if (start) begin
                busy  <= 1'b1;
                issue <= 1'b1;
                p     <= 13'd0;
                drain <= 3'd0;
            end else if (issue) begin
                if (p == plast) issue <= 1'b0;
                else            p <= p + 13'd1;
            end else if (busy && pipe_empty) begin
                // let the last write registers reach the banks
                drain <= drain + 3'd1;
                if (drain == 3'd3) busy <= 1'b0;
            end
        end
    end

endmodule
