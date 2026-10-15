// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

`timescale 1ns / 1ps

module uart_rx #(
    parameter CLK_FREQ = 100_000_000,
    parameter BAUD     = 3_000_000
)(
    input  wire       clk,
    input  wire       rst,
    input  wire       rx,
    output reg  [7:0] data_out,
    output reg        valid_out
);

    localparam CLKS_PER_BIT = CLK_FREQ / BAUD;

    reg [2:0] state = 3'b000;
    reg [15:0] clk_cnt = 16'd0;
    reg [2:0] bit_idx = 3'd0;
    reg [7:0] rx_byte = 8'd0;

    reg rx_sync1 = 1'b1;
    reg rx_sync2 = 1'b1;
    always @(posedge clk) begin
        rx_sync1 <= rx;
        rx_sync2 <= rx_sync1;
    end

    localparam S_IDLE  = 3'b000;
    localparam S_START = 3'b001;
    localparam S_DATA  = 3'b010;
    localparam S_STOP  = 3'b011;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state     <= S_IDLE;
            clk_cnt   <= 16'd0;
            bit_idx   <= 3'd0;
            rx_byte   <= 8'd0;
            data_out  <= 8'd0;
            valid_out <= 1'b0;
        end else begin
            valid_out <= 1'b0;
            case (state)
                S_IDLE: begin
                    clk_cnt <= 16'd0;
                    bit_idx <= 3'd0;
                    if (!rx_sync2) state <= S_START;
                end
                S_START: begin
                    if (clk_cnt == (CLKS_PER_BIT / 2)) begin
                        if (!rx_sync2) begin
                            clk_cnt <= 16'd0;
                            state   <= S_DATA;
                        end else begin
                            state <= S_IDLE;
                        end
                    end else begin
                        clk_cnt <= clk_cnt + 1'b1;
                    end
                end
                S_DATA: begin
                    if (clk_cnt < CLKS_PER_BIT - 1) begin
                        clk_cnt <= clk_cnt + 1'b1;
                    end else begin
                        clk_cnt <= 16'd0;
                        rx_byte[bit_idx] <= rx_sync2;
                        if (bit_idx < 3'd7) begin
                            bit_idx <= bit_idx + 1'b1;
                        end else begin
                            state <= S_STOP;
                        end
                    end
                end
                S_STOP: begin
                    if (clk_cnt < CLKS_PER_BIT - 1) begin
                        clk_cnt <= clk_cnt + 1'b1;
                    end else begin
                        state     <= S_IDLE;
                        data_out  <= rx_byte;
                        valid_out <= 1'b1;
                    end
                end
                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
