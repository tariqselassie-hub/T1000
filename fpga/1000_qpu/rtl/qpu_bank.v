// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

`timescale 1ns / 1ps

// One parity bank of the statevector: 8192 amplitudes of {im[31:0], re[31:0]}.
// Amplitude k lives in bank parity(k) at address k >> 1. One write and one read per cycle,
// registered read (data is valid the cycle after the address is applied).
module qpu_bank (
    input  wire        clk,
    input  wire        we,
    input  wire [12:0] wa,
    input  wire [63:0] wd,
    input  wire [12:0] ra,
    output reg  [63:0] rd
);

    (* ram_style = "block" *) reg [63:0] mem [0:8191];

    integer i;
    initial begin
        for (i = 0; i < 8192; i = i + 1) mem[i] = 64'd0;
        rd = 64'd0;
    end

    always @(posedge clk) begin
        if (we) mem[wa] <= wd;
        rd <= mem[ra];
    end

endmodule
