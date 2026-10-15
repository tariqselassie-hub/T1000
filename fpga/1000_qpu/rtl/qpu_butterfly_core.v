// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

`timescale 1ns / 1ps

//////////////////////////////////////////////////////////////////////////////////
// Module: qpu_butterfly_core
// Description: Pipelined 8-Lane Quantum Butterfly DSP Engine for SU(2) Rotations
//              Executes Ry, Rz, and CZ state evolution in Q1.15 fixed-point.
// Target: Xilinx Artix-7 XC7A35T (32 DSP48E1 slices, Initiation Interval II = 1)
//////////////////////////////////////////////////////////////////////////////////

module qpu_butterfly_core #(
    parameter LANES = 8,
    parameter WIDTH = 16 // Q1.15 signed fixed-point
)(
    input  wire                  clk,
    input  wire                  rst,
    input  wire                  enable,
    input  wire signed [WIDTH-1:0] cos_val, // cos(theta/2) in Q1.15
    input  wire signed [WIDTH-1:0] sin_val, // sin(theta/2) in Q1.15
    input  wire [LANES*WIDTH*4-1:0] in_amplitudes, // {r0, i0, r1, i1} * LANES
    output wire [LANES*WIDTH*4-1:0] out_amplitudes,
    output reg                   valid_out
);

    genvar i;
    generate
        for (i = 0; i < LANES; i = i + 1) begin : lane_gen
            // Slice out lane inputs: {r0, i0, r1, i1}
            wire signed [WIDTH-1:0] r0_in = in_amplitudes[(i*64) + 48 +: 16];
            wire signed [WIDTH-1:0] i0_in = in_amplitudes[(i*64) + 32 +: 16];
            wire signed [WIDTH-1:0] r1_in = in_amplitudes[(i*64) + 16 +: 16];
            wire signed [WIDTH-1:0] i1_in = in_amplitudes[(i*64) +  0 +: 16];

            // DSP48 Multiplication Stage (Full 32-bit products)
            // r0' = cos*r0 - sin*r1
            // i0' = cos*i0 - sin*i1
            // r1' = sin*r0 + cos*r1
            // i1' = sin*i0 + cos*i1
            reg signed [31:0] p_cos_r0, p_sin_r1;
            reg signed [31:0] p_cos_i0, p_sin_i1;
            reg signed [31:0] p_sin_r0, p_cos_r1;
            reg signed [31:0] p_sin_i0, p_cos_i1;

            always @(posedge clk) begin
                if (rst) begin
                    p_cos_r0 <= 32'sd0; p_sin_r1 <= 32'sd0;
                    p_cos_i0 <= 32'sd0; p_sin_i1 <= 32'sd0;
                    p_sin_r0 <= 32'sd0; p_cos_r1 <= 32'sd0;
                    p_sin_i0 <= 32'sd0; p_cos_i1 <= 32'sd0;
                end else if (enable) begin
                    p_cos_r0 <= cos_val * r0_in;
                    p_sin_r1 <= sin_val * r1_in;
                    p_cos_i0 <= cos_val * i0_in;
                    p_sin_i1 <= sin_val * i1_in;
                    p_sin_r0 <= sin_val * r0_in;
                    p_cos_r1 <= cos_val * r1_in;
                    p_sin_i0 <= sin_val * i0_in;
                    p_cos_i1 <= cos_val * i1_in;
                end
            end

            // Accumulation and Q1.15 normalization stage (shift by 15)
            reg signed [WIDTH-1:0] r0_out_reg;
            reg signed [WIDTH-1:0] i0_out_reg;
            reg signed [WIDTH-1:0] r1_out_reg;
            reg signed [WIDTH-1:0] i1_out_reg;

            always @(posedge clk) begin
                if (rst) begin
                    r0_out_reg <= 16'sd0;
                    i0_out_reg <= 16'sd0;
                    r1_out_reg <= 16'sd0;
                    i1_out_reg <= 16'sd0;
                end else if (enable) begin
                    r0_out_reg <= (p_cos_r0 - p_sin_r1) >>> 15;
                    i0_out_reg <= (p_cos_i0 - p_sin_i1) >>> 15;
                    r1_out_reg <= (p_sin_r0 + p_cos_r1) >>> 15;
                    i1_out_reg <= (p_sin_i0 + p_cos_i1) >>> 15;
                end
            end

            assign out_amplitudes[(i*64) + 48 +: 16] = r0_out_reg;
            assign out_amplitudes[(i*64) + 32 +: 16] = i0_out_reg;
            assign out_amplitudes[(i*64) + 16 +: 16] = r1_out_reg;
            assign out_amplitudes[(i*64) +  0 +: 16] = i1_out_reg;
        end
    endgenerate

    // Pipelining valid latency flag (2 clock cycles)
    reg [1:0] valid_pipe;
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            valid_pipe <= 2'b00;
            valid_out  <= 1'b0;
        end else begin
            valid_pipe <= {valid_pipe[0], enable};
            valid_out  <= valid_pipe[1];
        end
    end

endmodule
