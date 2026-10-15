// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

`timescale 1ns / 1ps

module uart_tx #(
    parameter CLK_FREQ = 100_000_000,
    parameter BAUD     = 3_000_000
)(
    input  wire       clk,
    input  wire       rst,
    input  wire       tx_start,
    input  wire [7:0] tx_data,
    output reg        tx,
    output wire       tx_busy
);

    localparam CLKS_PER_BIT = CLK_FREQ / BAUD;

    reg [2:0] state = 3'b000;
    reg [15:0] clk_cnt = 16'd0;
    reg [2:0] bit_idx = 3'd0;
    reg [7:0] tx_buf = 8'd0;

    localparam S_IDLE  = 3'b000;
    localparam S_START = 3'b001;
    localparam S_DATA  = 3'b010;
    localparam S_STOP  = 3'b011;

    assign tx_busy = (state != S_IDLE);

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state   <= S_IDLE;
            clk_cnt <= 16'd0;
            bit_idx <= 3'd0;
            tx      <= 1'b1;
            tx_buf  <= 8'd0;
        end else begin
            case (state)
                S_IDLE: begin
                    tx      <= 1'b1;
                    clk_cnt <= 16'd0;
                    bit_idx <= 3'd0;
                    if (tx_start) begin
                        tx_buf <= tx_data;
                        state  <= S_START;
                    end
                end
                S_START: begin
                    tx <= 1'b0; // Start bit
                    if (clk_cnt < CLKS_PER_BIT - 1) begin
                        clk_cnt <= clk_cnt + 1'b1;
                    end else begin
                        clk_cnt <= 16'd0;
                        state   <= S_DATA;
                    end
                end
                S_DATA: begin
                    tx <= tx_buf[bit_idx];
                    if (clk_cnt < CLKS_PER_BIT - 1) begin
                        clk_cnt <= clk_cnt + 1'b1;
                    end else begin
                        clk_cnt <= 16'd0;
                        if (bit_idx < 3'd7) begin
                            bit_idx <= bit_idx + 1'b1;
                        end else begin
                            state <= S_STOP;
                        end
                    end
                end
                S_STOP: begin
                    tx <= 1'b1; // Stop bit
                    if (clk_cnt < CLKS_PER_BIT - 1) begin
                        clk_cnt <= clk_cnt + 1'b1;
                    end else begin
                        clk_cnt <= 16'd0;
                        state   <= S_IDLE;
                    end
                end
                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
