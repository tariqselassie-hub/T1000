// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

`timescale 1ns / 1ps

//////////////////////////////////////////////////////////////////////////////////
// Module: qpu_seven_seg
// Description: Multiplexed 4-Digit 7-Segment Display Controller for Basys 3
//              Displays "1000" permanently to identify the 1000_QPU engine.
// Target: Xilinx Artix-7 XC7A35T-1CPG236C
//////////////////////////////////////////////////////////////////////////////////

module qpu_seven_seg (
    input  wire       clk_100mhz,
    input  wire       rst,
    output reg  [6:0] seg,     // Active LOW segment cathodes: {CA, CB, CC, CD, CE, CF, CG}
    output wire       dp,      // Active LOW decimal point
    output reg  [3:0] an       // Active LOW digit anodes: {AN3, AN2, AN1, AN0}
);

    // Refresh divider: 100 MHz / 2^17 ≈ 763 Hz refresh rate (~190 Hz per digit, flicker-free)
    reg [17:0] refresh_counter = 18'd0;
    always @(posedge clk_100mhz or posedge rst) begin
        if (rst)
            refresh_counter <= 18'd0;
        else
            refresh_counter <= refresh_counter + 1'b1;
    end

    wire [1:0] digit_sel = refresh_counter[17:16];

    // Segment definitions (Active LOW: 0 = Lit, 1 = Off)
    // Segments: {CG, CF, CE, CD, CC, CB, CA} -> seg[6:0]
    localparam SEG_0 = 7'b100_0000; // 0: A,B,C,D,E,F on, G off
    localparam SEG_1 = 7'b111_1001; // 1: B,C on, others off

    assign dp = 1'b1; // Decimal point off

    always @(*) begin
        case (digit_sel)
            2'b00: begin
                an  = 4'b1110;  // Digit 0 (Rightmost)
                seg = SEG_0;    // '0'
            end
            2'b01: begin
                an  = 4'b1101;  // Digit 1
                seg = SEG_0;    // '0'
            end
            2'b10: begin
                an  = 4'b1011;  // Digit 2
                seg = SEG_0;    // '0'
            end
            2'b11: begin
                an  = 4'b0111;  // Digit 3 (Leftmost)
                seg = SEG_1;    // '1'
            end
            default: begin
                an  = 4'b1111;
                seg = 7'b111_1111;
            end
        endcase
    end

endmodule
