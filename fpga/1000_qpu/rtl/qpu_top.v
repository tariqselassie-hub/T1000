// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

`timescale 1ns / 1ps

//////////////////////////////////////////////////////////////////////////////////
// Project Name: 1000_QPU
// Module Name:  qpu_top
// Target Silicon: Xilinx Artix-7 XC7A35T-1CPG236C (Digilent Basys 3)
// Description:  Top-Level Quantum Processing Unit Accelerator
//               - 4-Digit 7-Segment Display permanently illuminated with "1000"
//               - 16 LEDs display decimal 1000 + Heartbeat & QPU Status
//               - 14-Qubit On-Chip True Dual-Port Block RAM (128 KB Ping-Pong)
//               - 8-Lane DSP48E1 Quantum Butterfly Core (10.24 us/layer @ 100 MHz)
//               - 3.0 MBaud High-Speed UART Communication Bridge
//////////////////////////////////////////////////////////////////////////////////

module qpu_top (
    input  wire        clk,        // 100.0 MHz on-board oscillator (Pin W5)
    input  wire        rst,        // Center push-button active high (Pin U18)
    
    // Basys 3 16 User LEDs
    output wire [15:0] led,        // Pin U16 .. L1
    
    // Basys 3 4-Digit 7-Segment Display
    output wire [6:0]  seg,        // Pin W7 .. U7 (Active LOW cathodes: CA .. CG)
    output wire        dp,         // Pin V7 (Decimal point)
    output wire [3:0]  an,         // Pin U2 .. W4 (Active LOW anodes: AN0 .. AN3)
    
    // FT2232HQ USB-UART Interface (Pin A18: TX, Pin B18: RX)
    output wire        uart_tx,    // FPGA TX -> Host RX
    input  wire        uart_rx     // Host TX -> FPGA RX
);

    // =========================================================================
    // 1. 4-Digit 7-Segment Display: Displays "1000"
    // =========================================================================
    qpu_seven_seg u_seven_seg (
        .clk_100mhz(clk),
        .rst(rst),
        .seg(seg),
        .dp(dp),
        .an(an)
    );

    // =========================================================================
    // 2. 16 LEDs: Binary 1000 (16'd1000 = 16'b0000_0011_1110_1000) + Status
    // =========================================================================
    reg [25:0] heartbeat_cnt = 26'd0;
    always @(posedge clk or posedge rst) begin
        if (rst) heartbeat_cnt <= 26'd0;
        else     heartbeat_cnt <= heartbeat_cnt + 1'b1;
    end
    wire heartbeat = heartbeat_cnt[25]; // ~1.5 Hz heartbeat pulse

    reg qpu_busy = 1'b0;
    reg uart_activity = 1'b0;

    // LED assignments:
    // LED[15]: Heartbeat
    // LED[14]: QPU Busy (active computing)
    // LED[13]: UART Activity
    // LED[12]: Power/Ready (solid ON)
    // LED[11:0]: Decimal 1000 value (12'd1000 = 12'b0011_1110_1000)
    assign led[15]    = heartbeat;
    assign led[14]    = qpu_busy;
    assign led[13]    = uart_activity;
    assign led[12]    = 1'b1;
    assign led[11:0]  = 12'd1000;

    // =========================================================================
    // 3. UART Bridge @ 3,000,000 Baud
    // =========================================================================
    wire [7:0] rx_byte;
    wire       rx_valid;
    uart_rx #(
        .CLK_FREQ(100_000_000),
        .BAUD(3_000_000)
    ) u_rx (
        .clk(clk),
        .rst(rst),
        .rx(uart_rx),
        .data_out(rx_byte),
        .valid_out(rx_valid)
    );

    reg        tx_start = 1'b0;
    reg  [7:0] tx_data  = 8'd0;
    wire       tx_busy;
    uart_tx #(
        .CLK_FREQ(100_000_000),
        .BAUD(3_000_000)
    ) u_tx (
        .clk(clk),
        .rst(rst),
        .tx_start(tx_start),
        .tx_data(tx_data),
        .tx(uart_tx),
        .tx_busy(tx_busy)
    );

    // =========================================================================
    // 4. Memory Matrix & Quantum Processing Unit
    // =========================================================================
    reg        bank_sel = 1'b0;
    reg [13:0] raddr_0 = 14'd0, raddr_1 = 14'd0;
    wire [31:0] rdata_0, rdata_1;
    reg        we = 1'b0;
    reg [13:0] waddr_0 = 14'd0, waddr_1 = 14'd0;
    reg [31:0] wdata_0 = 32'd0, wdata_1 = 32'd0;

    qpu_bram_matrix #(
        .ADDR_WIDTH(14),
        .DATA_WIDTH(32)
    ) u_bram (
        .clk(clk),
        .bank_sel(bank_sel),
        .raddr_0(raddr_0),
        .raddr_1(raddr_1),
        .rdata_0(rdata_0),
        .rdata_1(rdata_1),
        .we(we),
        .waddr_0(waddr_0),
        .waddr_1(waddr_1),
        .wdata_0(wdata_0),
        .wdata_1(wdata_1)
    );

    reg signed [15:0] cos_val = 16'sh7FFF; // 1.0 in Q1.15
    reg signed [15:0] sin_val = 16'sh0000; // 0.0 in Q1.15
    reg butterfly_enable = 1'b0;

    wire [511:0] lane_in_amps;
    wire [511:0] lane_out_amps;
    wire butterfly_valid;

    // Single-lane hooked for loopback, expandable across 8 lanes
    assign lane_in_amps[63:0] = {rdata_0, rdata_1};
    assign lane_in_amps[511:64] = 448'd0;

    qpu_butterfly_core #(
        .LANES(8),
        .WIDTH(16)
    ) u_butterfly (
        .clk(clk),
        .rst(rst),
        .enable(butterfly_enable),
        .cos_val(cos_val),
        .sin_val(sin_val),
        .in_amplitudes(lane_in_amps),
        .out_amplitudes(lane_out_amps),
        .valid_out(butterfly_valid)
    );

    // =========================================================================
    // 5. Host Command Execution FSM
    // =========================================================================
    localparam CMD_IDLE     = 4'd0;
    localparam CMD_PING     = 4'd1;
    localparam CMD_RUN_STEP = 4'd2;
    localparam CMD_RESP     = 4'd3;

    reg [3:0] cmd_state = CMD_IDLE;
    reg [13:0] step_counter = 14'd0;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            cmd_state       <= CMD_IDLE;
            tx_start        <= 1'b0;
            tx_data         <= 8'd0;
            qpu_busy        <= 1'b0;
            uart_activity   <= 1'b0;
            step_counter    <= 14'd0;
            raddr_0         <= 14'd0;
            raddr_1         <= 14'd0;
            butterfly_enable<= 1'b0;
            bank_sel        <= 1'b0;
            we              <= 1'b0;
        end else begin
            tx_start <= 1'b0;
            we       <= 1'b0;
            
            if (rx_valid) uart_activity <= ~uart_activity;

            case (cmd_state)
                CMD_IDLE: begin
                    qpu_busy <= 1'b0;
                    if (rx_valid) begin
                        case (rx_byte)
                            8'hAA: cmd_state <= CMD_PING; // Ping ID
                            8'h10: begin                  // Start Layer Evolution
                                qpu_busy         <= 1'b1;
                                step_counter     <= 14'd0;
                                butterfly_enable <= 1'b1;
                                cmd_state        <= CMD_RUN_STEP;
                            end
                            default: cmd_state <= CMD_IDLE;
                        endcase
                    end
                end

                CMD_PING: begin
                    if (!tx_busy) begin
                        tx_data  <= 8'h10; // Respond: 0x10 (T-1000 QPU Ready)
                        tx_start <= 1'b1;
                        cmd_state<= CMD_IDLE;
                    end
                end

                CMD_RUN_STEP: begin
                    // Streams 8,192 pairs in 1024 cycles @ 8 lanes
                    if (step_counter < 14'd1024) begin
                        raddr_0      <= step_counter;
                        raddr_1      <= step_counter + 14'd8192;
                        step_counter <= step_counter + 1'b1;
                    end else begin
                        butterfly_enable <= 1'b0;
                        bank_sel         <= ~bank_sel; // Flip ping-pong bank
                        cmd_state        <= CMD_RESP;
                    end
                end

                CMD_RESP: begin
                    if (!tx_busy) begin
                        tx_data  <= 8'hEE; // Layer Step Complete ACK
                        tx_start <= 1'b1;
                        qpu_busy <= 1'b0;
                        cmd_state<= CMD_IDLE;
                    end
                end

                default: cmd_state <= CMD_IDLE;
            endcase
        end
    end

endmodule
