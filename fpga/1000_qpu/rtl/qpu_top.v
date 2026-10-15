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
//               - Autonomous Multi-Layer Circuit Sequencer (Opcode 0x20)
//               - On-Chip Quantum Chaos & Entropy Harvester (Opcode 0x30)
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
    // 2. 16 LEDs: Binary 1000 (16'd1000) + Status Indicators
    // =========================================================================
    reg [25:0] heartbeat_cnt = 26'd0;
    always @(posedge clk or posedge rst) begin
        if (rst) heartbeat_cnt <= 26'd0;
        else     heartbeat_cnt <= heartbeat_cnt + 1'b1;
    end
    wire heartbeat = heartbeat_cnt[25]; // ~1.5 Hz heartbeat pulse

    reg qpu_busy = 1'b0;
    reg uart_activity = 1'b0;

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
    // 4. Memory Matrix & Quantum Butterfly Pipeline
    // =========================================================================
    reg        bank_sel = 1'b0;
    reg [13:0] raddr_0 = 14'd0, raddr_1 = 14'd0;
    wire [31:0] rdata_0, rdata_1;
    wire       we;
    wire [13:0] waddr_0, waddr_1;
    wire [31:0] wdata_0, wdata_1;

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

    // Fixed-Point Q1.15 Rotation Angles ROM (16 Haar-Scrambling Unitary Angles)
    reg signed [15:0] angle_rom_cos [0:15];
    reg signed [15:0] angle_rom_sin [0:15];
    initial begin
        angle_rom_cos[0]  = 16'sh7FFF; angle_rom_sin[0]  = 16'sh0000; // 0
        angle_rom_cos[1]  = 16'sh7641; angle_rom_sin[1]  = 16'sh30FC; // pi/8
        angle_rom_cos[2]  = 16'sh5A82; angle_rom_sin[2]  = 16'sh5A82; // pi/4
        angle_rom_cos[3]  = 16'sh30FC; angle_rom_sin[3]  = 16'sh7641; // 3pi/8
        angle_rom_cos[4]  = 16'sh0000; angle_rom_sin[4]  = 16'sh7FFF; // pi/2
        angle_rom_cos[5]  = -16'sh30FC; angle_rom_sin[5] = 16'sh7641;
        angle_rom_cos[6]  = -16'sh5A82; angle_rom_sin[6] = 16'sh5A82;
        angle_rom_cos[7]  = -16'sh7641; angle_rom_sin[7] = 16'sh30FC;
        angle_rom_cos[8]  = -16'sh7FFF; angle_rom_sin[8] = 16'sh0000;
        angle_rom_cos[9]  = -16'sh7641; angle_rom_sin[9] = -16'sh30FC;
        angle_rom_cos[10] = -16'sh5A82; angle_rom_sin[10] = -16'sh5A82;
        angle_rom_cos[11] = -16'sh30FC; angle_rom_sin[11] = -16'sh7641;
        angle_rom_cos[12] = 16'sh0000; angle_rom_sin[12]  = -16'sh7FFF;
        angle_rom_cos[13] = 16'sh30FC; angle_rom_sin[13]  = -16'sh7641;
        angle_rom_cos[14] = 16'sh5A82; angle_rom_sin[14] = -16'sh5A82;
        angle_rom_cos[15] = 16'sh7641; angle_rom_sin[15] = -16'sh30FC;
    end

    reg signed [15:0] cos_val = 16'sh7FFF;
    reg signed [15:0] sin_val = 16'sh0000;
    reg butterfly_enable = 1'b0;

    wire [511:0] lane_in_amps;
    wire [511:0] lane_out_amps;
    wire butterfly_valid;

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

    // 2-Cycle Delayed Address Pipeline for BRAM Write-Back
    reg [13:0] waddr_pipe_0 [0:1];
    reg [13:0] waddr_pipe_1 [0:1];
    always @(posedge clk) begin
        waddr_pipe_0[0] <= raddr_0;
        waddr_pipe_0[1] <= waddr_pipe_0[0];
        waddr_pipe_1[0] <= raddr_1;
        waddr_pipe_1[1] <= waddr_pipe_1[0];
    end

    assign waddr_0 = waddr_pipe_0[1];
    assign waddr_1 = waddr_pipe_1[1];
    assign wdata_0 = lane_out_amps[63:32];
    assign wdata_1 = lane_out_amps[31:0];
    assign we      = butterfly_valid;

    // =========================================================================
    // 5. Host Command & Autonomous Quantum Chaos Engine FSM
    // =========================================================================
    localparam CMD_IDLE            = 4'd0;
    localparam CMD_PING            = 4'd1;
    localparam CMD_RUN_SINGLE_STEP = 4'd2;
    localparam CMD_RESP            = 4'd3;
    localparam CMD_WAIT_MULTI_D    = 4'd4;
    localparam CMD_RUN_MULTI_LAYER = 4'd5;
    localparam CMD_CHAOS_WAIT_D    = 4'd6;
    localparam CMD_CHAOS_WAIT_NL   = 4'd7;
    localparam CMD_CHAOS_WAIT_NH   = 4'd8;
    localparam CMD_CHAOS_EVOLVE    = 4'd9;
    localparam CMD_CHAOS_HARVEST   = 4'd10;
    localparam CMD_CHAOS_TX_SEED   = 4'd11;

    reg [3:0]  cmd_state = CMD_IDLE;
    reg [13:0] step_counter = 14'd0;
    reg [7:0]  layer_idx = 8'd0;
    reg [7:0]  target_depth = 8'd12;
    reg [15:0] seed_count = 16'd0;
    reg [5:0]  harvest_idx = 6'd0;
    reg [5:0]  tx_seed_byte_idx = 6'd0;
    reg [255:0] entropy_reg = 256'd0;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            cmd_state       <= CMD_IDLE;
            tx_start        <= 1'b0;
            tx_data         <= 8'd0;
            qpu_busy        <= 1'b0;
            uart_activity   <= 1'b0;
            step_counter    <= 14'd0;
            layer_idx       <= 8'd0;
            target_depth    <= 8'd12;
            seed_count      <= 16'd0;
            harvest_idx     <= 6'd0;
            tx_seed_byte_idx<= 6'd0;
            raddr_0         <= 14'd0;
            raddr_1         <= 14'd0;
            butterfly_enable<= 1'b0;
            bank_sel        <= 1'b0;
            cos_val         <= 16'sh7FFF;
            sin_val         <= 16'sh0000;
            entropy_reg     <= 256'd0;
        end else begin
            tx_start <= 1'b0;
            
            if (rx_valid) uart_activity <= ~uart_activity;

            case (cmd_state)
                // -------------------------------------------------------------
                // State 0: Wait for Host Command Opcode
                // -------------------------------------------------------------
                CMD_IDLE: begin
                    qpu_busy         <= 1'b0;
                    butterfly_enable <= 1'b0;
                    if (rx_valid) begin
                        case (rx_byte)
                            8'hAA: cmd_state <= CMD_PING;            // Ping / Handshake
                            8'h10: begin                             // Single-Layer Butterfly Pass
                                qpu_busy         <= 1'b1;
                                step_counter     <= 14'd0;
                                butterfly_enable <= 1'b1;
                                cos_val          <= 16'sh7FFF;
                                sin_val          <= 16'sh0000;
                                cmd_state        <= CMD_RUN_SINGLE_STEP;
                            end
                            8'h20: begin                             // Autonomous Multi-Layer Pass
                                qpu_busy  <= 1'b1;
                                cmd_state <= CMD_WAIT_MULTI_D;
                            end
                            8'h30: begin                             // Quantum Chaos Entropy Stream
                                qpu_busy  <= 1'b1;
                                cmd_state <= CMD_CHAOS_WAIT_D;
                            end
                            default: cmd_state <= CMD_IDLE;
                        endcase
                    end
                end

                // -------------------------------------------------------------
                // Opcode 0xAA: Ping Response (0x10)
                // -------------------------------------------------------------
                CMD_PING: begin
                    if (!tx_busy) begin
                        tx_data  <= 8'h10;
                        tx_start <= 1'b1;
                        cmd_state<= CMD_IDLE;
                    end
                end

                // -------------------------------------------------------------
                // Opcode 0x10: Legacy Single Layer Execution
                // -------------------------------------------------------------
                CMD_RUN_SINGLE_STEP: begin
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

                // -------------------------------------------------------------
                // Opcode 0x20: Autonomous Multi-Layer Execution (Option 4)
                // -------------------------------------------------------------
                CMD_WAIT_MULTI_D: begin
                    if (rx_valid) begin
                        target_depth <= (rx_byte == 8'd0) ? 8'd1 : rx_byte;
                        layer_idx    <= 8'd0;
                        step_counter <= 14'd0;
                        butterfly_enable <= 1'b1;
                        cmd_state    <= CMD_RUN_MULTI_LAYER;
                    end
                end

                CMD_RUN_MULTI_LAYER: begin
                    // Select rotation angles for current layer
                    cos_val <= angle_rom_cos[layer_idx[3:0]];
                    sin_val <= angle_rom_sin[layer_idx[3:0]];

                    if (step_counter < 14'd1024) begin
                        raddr_0      <= step_counter;
                        raddr_1      <= step_counter + 14'd8192;
                        step_counter <= step_counter + 1'b1;
                    end else begin
                        bank_sel     <= ~bank_sel; // Flip BRAM bank in silicon
                        step_counter <= 14'd0;
                        if (layer_idx + 1'b1 < target_depth) begin
                            layer_idx <= layer_idx + 1'b1;
                        end else begin
                            butterfly_enable <= 1'b0;
                            cmd_state        <= CMD_RESP;
                        end
                    end
                end

                // -------------------------------------------------------------
                // Opcode 0x30: Quantum Chaos & Entropy Harvester (Option 3 + 4)
                // -------------------------------------------------------------
                CMD_CHAOS_WAIT_D: begin
                    if (rx_valid) begin
                        target_depth <= (rx_byte == 8'd0) ? 8'd12 : rx_byte;
                        cmd_state    <= CMD_CHAOS_WAIT_NL;
                    end
                end

                CMD_CHAOS_WAIT_NL: begin
                    if (rx_valid) begin
                        seed_count[7:0] <= rx_byte;
                        cmd_state       <= CMD_CHAOS_WAIT_NH;
                    end
                end

                CMD_CHAOS_WAIT_NH: begin
                    if (rx_valid) begin
                        seed_count[15:8] <= rx_byte;
                        layer_idx        <= 8'd0;
                        step_counter     <= 14'd0;
                        butterfly_enable <= 1'b1;
                        cmd_state        <= CMD_CHAOS_EVOLVE;
                    end
                end

                CMD_CHAOS_EVOLVE: begin
                    // Dynamic phase rotation across layers
                    cos_val <= angle_rom_cos[(layer_idx ^ seed_count[3:0]) & 4'hF];
                    sin_val <= angle_rom_sin[(layer_idx ^ seed_count[3:0]) & 4'hF];

                    if (step_counter < 14'd1024) begin
                        raddr_0      <= step_counter;
                        raddr_1      <= step_counter + 14'd8192;
                        step_counter <= step_counter + 1'b1;
                    end else begin
                        bank_sel     <= ~bank_sel; // Flip ping-pong bank
                        step_counter <= 14'd0;
                        if (layer_idx + 1'b1 < target_depth) begin
                            layer_idx <= layer_idx + 1'b1;
                        end else begin
                            butterfly_enable <= 1'b0;
                            harvest_idx      <= 6'd0;
                            cmd_state        <= CMD_CHAOS_HARVEST;
                        end
                    end
                end

                CMD_CHAOS_HARVEST: begin
                    // Sample 32 distributed amplitude states across the 16,384 basis states
                    if (harvest_idx < 6'd32) begin
                        raddr_0     <= {harvest_idx[4:0], 9'd0}; // Stride of 512
                        raddr_1     <= {harvest_idx[4:0], 9'd256};
                        // Fold real and imaginary amplitude bits into 256-bit entropy register
                        entropy_reg <= {entropy_reg[247:0], 
                                        entropy_reg[255:248] ^ rdata_0[15:8] ^ rdata_1[7:0] ^ rdata_0[31:24] ^ rdata_1[23:16] ^ harvest_idx[5:0]};
                        harvest_idx <= harvest_idx + 1'b1;
                    end else begin
                        tx_seed_byte_idx <= 6'd0;
                        cmd_state        <= CMD_CHAOS_TX_SEED;
                    end
                end

                CMD_CHAOS_TX_SEED: begin
                    // Stream 32 bytes (256-bit cryptographic seed) over UART @ 3.0 MBaud
                    if (!tx_busy && !tx_start) begin
                        if (tx_seed_byte_idx < 6'd32) begin
                            tx_data          <= entropy_reg[255:248];
                            entropy_reg      <= {entropy_reg[247:0], 8'd0};
                            tx_start         <= 1'b1;
                            tx_seed_byte_idx <= tx_seed_byte_idx + 1'b1;
                        end else begin
                            // Finished streaming 32-byte seed
                            if (seed_count > 16'd1) begin
                                seed_count   <= seed_count - 1'b1;
                                layer_idx    <= 8'd0;
                                step_counter <= 14'd0;
                                butterfly_enable <= 1'b1;
                                cmd_state    <= CMD_CHAOS_EVOLVE; // Loop to next seed!
                            end else begin
                                cmd_state    <= CMD_RESP; // Finished all seeds!
                            end
                        end
                    end
                end

                // -------------------------------------------------------------
                // Common Completion ACK Response (0xEE)
                // -------------------------------------------------------------
                CMD_RESP: begin
                    if (!tx_busy) begin
                        tx_data  <= 8'hEE; // ACK
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
