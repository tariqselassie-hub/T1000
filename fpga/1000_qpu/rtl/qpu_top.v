// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

`timescale 1ns / 1ps

//////////////////////////////////////////////////////////////////////////////////
// Project Name: 1000_QPU
// Module Name:  qpu_top
// Target:       Xilinx Artix-7 XC7A35T-1CPG236C (Digilent Basys 3)
// Description:  Fixed-point statevector engine for up to 14 qubits.
//               - 2 x 8192 x 64-bit block-RAM banks (amplitude k in bank parity(k), address k>>1)
//               - controlled 2x2 complex gate, one amplitude pair per clock
//               - load / read back / reset / sum of Z-string probabilities over UART
//               The byte protocol and the arithmetic are specified by the Rust model in
//               crates/t1000/src/qpu/{protocol,model,fixed}.rs; the testbench replays the
//               vectors that model produces.
//////////////////////////////////////////////////////////////////////////////////

module qpu_top #(
    parameter CLK_FREQ = 100_000_000,
    parameter BAUD     = 3_000_000
)(
    input  wire        clk,        // 100 MHz oscillator (W5)
    input  wire        rst,        // centre push-button, active high (U18)
    output wire [15:0] led,
    output wire [6:0]  seg,
    output wire        dp,
    output wire [3:0]  an,
    output wire        uart_tx,    // FPGA TX -> host RX (A18)
    input  wire        uart_rx     // host TX -> FPGA RX (B18)
);

    // ---- display and status LEDs -------------------------------------------------
    qpu_seven_seg u_seven_seg (.clk_100mhz(clk), .rst(rst), .seg(seg), .dp(dp), .an(an));

    reg [25:0] heartbeat_cnt = 26'd0;
    always @(posedge clk or posedge rst) begin
        if (rst) heartbeat_cnt <= 26'd0;
        else     heartbeat_cnt <= heartbeat_cnt + 1'b1;
    end

    // ---- UART -------------------------------------------------------------------
    wire [7:0] rx_byte;
    wire       rx_valid;
    uart_rx #(.CLK_FREQ(CLK_FREQ), .BAUD(BAUD)) u_rx (
        .clk(clk), .rst(rst), .rx(uart_rx), .data_out(rx_byte), .valid_out(rx_valid)
    );

    reg        tx_start = 1'b0;
    reg  [7:0] tx_data  = 8'd0;
    wire       tx_busy;
    uart_tx #(.CLK_FREQ(CLK_FREQ), .BAUD(BAUD)) u_tx (
        .clk(clk), .rst(rst), .tx_start(tx_start), .tx_data(tx_data), .tx(uart_tx), .tx_busy(tx_busy)
    );

    // ---- receive FIFO (512 bytes) -----------------------------------------------
    reg [7:0] fifo [0:511];
    reg [8:0] wptr = 9'd0;
    reg [8:0] rptr = 9'd0;
    reg       overflow = 1'b0;
    wire      have = (wptr != rptr);
    wire      full = ((wptr + 9'd1) == rptr);
    wire [7:0] rxb = fifo[rptr];

    always @(posedge clk) begin
        if (rx_valid && !full) fifo[wptr] <= rx_byte;
    end

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            wptr <= 9'd0;
            overflow <= 1'b0;
        end else if (rx_valid) begin
            if (full) overflow <= 1'b1;
            else      wptr <= wptr + 9'd1;
        end
    end

    // ---- banks ------------------------------------------------------------------
    reg         h_we0 = 1'b0, h_we1 = 1'b0;
    reg  [12:0] h_wa0 = 13'd0, h_wa1 = 13'd0, h_ra = 13'd0;
    reg  [63:0] h_wd0 = 64'd0, h_wd1 = 64'd0;

    wire        g_busy;
    wire [12:0] g_ra0, g_ra1, g_wa0, g_wa1;
    wire [63:0] g_wd0, g_wd1;
    wire        g_we0, g_we1;
    wire [63:0] rd0, rd1;

    wire        we0 = g_busy ? g_we0 : h_we0;
    wire        we1 = g_busy ? g_we1 : h_we1;
    wire [12:0] wa0 = g_busy ? g_wa0 : h_wa0;
    wire [12:0] wa1 = g_busy ? g_wa1 : h_wa1;
    wire [63:0] wd0 = g_busy ? g_wd0 : h_wd0;
    wire [63:0] wd1 = g_busy ? g_wd1 : h_wd1;
    wire [12:0] ra0 = g_busy ? g_ra0 : h_ra;
    wire [12:0] ra1 = g_busy ? g_ra1 : h_ra;

    qpu_bank u_bank0 (.clk(clk), .we(we0), .wa(wa0), .wd(wd0), .ra(ra0), .rd(rd0));
    qpu_bank u_bank1 (.clk(clk), .we(we1), .wa(wa1), .wd(wd1), .ra(ra1), .rd(rd1));

    // ---- command state ----------------------------------------------------------
    localparam [7:0] OP_PING = 8'hAA, OP_VERSION = 8'h01, OP_SET_N = 8'h02, OP_RESET = 8'h03,
                     OP_LOAD = 8'h04, OP_READ = 8'h05, OP_GATE = 8'h06, OP_EXPECT = 8'h07,
                     OP_LOAD_CIRCUIT = 8'h08, OP_RUN_BURST = 8'h09, OP_RUN_BURST_EXPECT = 8'h0A;
    localparam [7:0] ACK = 8'hEE, ERR = 8'hE1, UNKNOWN = 8'hE0;

    localparam [5:0]
        S_IDLE = 6'd0,  S_ARGS = 6'd1,  S_EXEC = 6'd2,  S_OUT = 6'd3,   S_OUTW = 6'd4,
        S_ACK = 6'd5,   S_ERR = 6'd6,   S_RST = 6'd7,   S_LDB = 6'd8,   S_LDW = 6'd9,
        S_RD = 6'd10,   S_RD1 = 6'd11,  S_RD2 = 6'd12,  S_GST = 6'd13,  S_GWAIT = 6'd14,
        S_EZ = 6'd15,   S_EZ1 = 6'd16,  S_EZ2 = 6'd17,  S_EZ3 = 6'd18,  S_EZ4 = 6'd19,
        S_EZW = 6'd20,  S_LC_BYTE = 6'd21, S_BST_ST = 6'd22, S_BST_WAIT = 6'd23;

    reg [5:0]  state    = S_IDLE;
    reg [5:0]  out_next = S_IDLE;
    reg [7:0]  op       = 8'd0;
    reg [7:0]  args [0:35];
    reg [5:0]  acnt     = 6'd0;
    reg [5:0]  anum     = 6'd0;
    reg [3:0]  n        = 4'd14;
    reg [79:0] txsr     = 80'd0;
    reg [3:0]  txn      = 4'd0;

    reg [14:0] k        = 15'd0;
    reg [15:0] remain   = 16'd0;
    reg        ld_ok    = 1'b0;
    reg [63:0] amp      = 64'd0;
    reg [3:0]  ldb      = 4'd0;
    reg [12:0] j        = 13'd0;
    reg        gate_start = 1'b0;

    // ---- on-chip circuit microcode memory (up to 128 gates) ---------------------
    localparam MAX_CIRC_GATES = 128;
    reg [7:0]   cm_target [0:MAX_CIRC_GATES-1];
    reg [15:0]  cm_ctrl   [0:MAX_CIRC_GATES-1];
    reg [31:0]  cm_m00r   [0:MAX_CIRC_GATES-1];
    reg [31:0]  cm_m00i   [0:MAX_CIRC_GATES-1];
    reg [31:0]  cm_m01r   [0:MAX_CIRC_GATES-1];
    reg [31:0]  cm_m01i   [0:MAX_CIRC_GATES-1];
    reg [31:0]  cm_m10r   [0:MAX_CIRC_GATES-1];
    reg [31:0]  cm_m10i   [0:MAX_CIRC_GATES-1];
    reg [31:0]  cm_m11r   [0:MAX_CIRC_GATES-1];
    reg [31:0]  cm_m11i   [0:MAX_CIRC_GATES-1];
    reg [7:0]   cm_wptr   = 8'd0;
    reg [7:0]   cm_rptr   = 8'd0;
    reg [7:0]   cm_len    = 8'd0;
    reg         burst_mode = 1'b0;
    reg         expect_after_burst = 1'b0;
    reg [15:0]  burst_mask = 16'd0;
    reg [7:0]   g_buf [0:34];
    reg [5:0]   gb_idx = 6'd0;

    // expect-Z datapath
    reg signed [31:0] ez_re = 32'sd0, ez_im = 32'sd0;
    reg        [63:0] ez_pr = 64'd0, ez_pi = 64'd0;
    reg signed [79:0] ez_acc = 80'sd0;
    reg        [13:0] ez_mask = 14'd0;

    wire [14:0] dim     = 15'd1 << n;
    wire [12:0] half_m1 = (13'd1 << (n - 4'd1)) - 13'd1;
    wire        kpar    = ^k[13:0];

    // argument views (args[] holds the bytes after the opcode)
    wire [15:0] a_addr   = {args[1], args[0]};
    wire [15:0] a_cnt    = {args[3], args[2]};
    wire [16:0] a_end    = {1'b0, a_addr} + {1'b0, a_cnt};
    wire        a_fits   = (a_end <= {2'b00, dim});
    wire [31:0] a_one    = {args[3], args[2], args[1], args[0]};
    wire [15:0] g_ctrl16 = {args[2], args[1]};
    wire        g_ok     = (args[0] < {4'd0, n}) && ((g_ctrl16 >> n) == 16'd0)
                           && !g_ctrl16[args[0][3:0]];
    wire [15:0] z_mask16 = {args[1], args[0]};
    wire [15:0] circ_cnt = {args[1], args[0]};

    // gate: args[0] target, args[1..2] control mask, args[3..34] eight little-endian words
    wire signed [24:0] c00r = {args[6][0],  args[5],  args[4],  args[3]};
    wire signed [24:0] c00i = {args[10][0], args[9],  args[8],  args[7]};
    wire signed [24:0] c01r = {args[14][0], args[13], args[12], args[11]};
    wire signed [24:0] c01i = {args[18][0], args[17], args[16], args[15]};
    wire signed [24:0] c10r = {args[22][0], args[21], args[20], args[19]};
    wire signed [24:0] c10i = {args[26][0], args[25], args[24], args[23]};
    wire signed [24:0] c11r = {args[30][0], args[29], args[28], args[27]};
    wire signed [24:0] c11i = {args[34][0], args[33], args[32], args[31]};

    // gate multiplexer between single UART gate and burst microcode memory
    wire [7:0]  cur_target = burst_mode ? cm_target[cm_rptr]          : args[0];
    wire [15:0] cur_ctrl16 = burst_mode ? cm_ctrl[cm_rptr]            : g_ctrl16;
    wire signed [24:0] cur_m00r = burst_mode ? cm_m00r[cm_rptr][24:0] : c00r;
    wire signed [24:0] cur_m00i = burst_mode ? cm_m00i[cm_rptr][24:0] : c00i;
    wire signed [24:0] cur_m01r = burst_mode ? cm_m01r[cm_rptr][24:0] : c01r;
    wire signed [24:0] cur_m01i = burst_mode ? cm_m01i[cm_rptr][24:0] : c01i;
    wire signed [24:0] cur_m10r = burst_mode ? cm_m10r[cm_rptr][24:0] : c10r;
    wire signed [24:0] cur_m10i = burst_mode ? cm_m10i[cm_rptr][24:0] : c10i;
    wire signed [24:0] cur_m11r = burst_mode ? cm_m11r[cm_rptr][24:0] : c11r;
    wire signed [24:0] cur_m11i = burst_mode ? cm_m11i[cm_rptr][24:0] : c11i;
    wire cur_g_ok = (cur_target < {4'd0, n}) && ((cur_ctrl16 >> n) == 16'd0)
                    && !cur_ctrl16[cur_target[3:0]];

    qpu_gate u_gate (
        .clk(clk), .rst(rst), .start(gate_start), .n(n), .target(cur_target[3:0]),
        .ctrl(cur_ctrl16[13:0]),
        .m00r(cur_m00r), .m00i(cur_m00i), .m01r(cur_m01r), .m01i(cur_m01i),
        .m10r(cur_m10r), .m10i(cur_m10i), .m11r(cur_m11r), .m11i(cur_m11i),
        .busy(g_busy), .ra0(g_ra0), .ra1(g_ra1), .rd0(rd0), .rd1(rd1),
        .we0(g_we0), .we1(g_we1), .wa0(g_wa0), .wa1(g_wa1), .wd0(g_wd0), .wd1(g_wd1)
    );

    wire [63:0] k_amp = kpar ? rd1 : rd0;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= S_IDLE; out_next <= S_IDLE; n <= 4'd14;
            tx_start <= 1'b0; gate_start <= 1'b0; h_we0 <= 1'b0; h_we1 <= 1'b0;
            rptr <= 9'd0; txn <= 4'd0;
            burst_mode <= 1'b0; cm_len <= 8'd0; cm_wptr <= 8'd0; cm_rptr <= 8'd0;
            gb_idx <= 6'd0; expect_after_burst <= 1'b0;
        end else begin
            tx_start   <= 1'b0;
            gate_start <= 1'b0;
            h_we0      <= 1'b0;
            h_we1      <= 1'b0;

            case (state)
                // ---- command intake ------------------------------------------------
                S_IDLE: if (have) begin
                    rptr <= rptr + 9'd1;
                    op   <= rxb;
                    acnt <= 6'd0;
                    case (rxb)
                        OP_PING:             begin txsr <= 80'h10; txn <= 4'd1; out_next <= S_IDLE; state <= S_OUT; end
                        OP_VERSION:          begin txsr <= 80'h01_55_50_51; txn <= 4'd4; out_next <= S_ACK; state <= S_OUT; end
                        OP_SET_N:            begin anum <= 6'd1;  state <= S_ARGS; end
                        OP_RESET:            begin anum <= 6'd4;  state <= S_ARGS; end
                        OP_LOAD:             begin anum <= 6'd4;  state <= S_ARGS; end
                        OP_READ:             begin anum <= 6'd4;  state <= S_ARGS; end
                        OP_GATE:             begin anum <= 6'd35; state <= S_ARGS; end
                        OP_EXPECT:           begin anum <= 6'd2;  state <= S_ARGS; end
                        OP_LOAD_CIRCUIT:     begin anum <= 6'd2;  state <= S_ARGS; end
                        OP_RUN_BURST:        begin
                            if (cm_len > 8'd0) begin
                                cm_rptr <= 8'd0;
                                burst_mode <= 1'b1;
                                expect_after_burst <= 1'b0;
                                state <= S_BST_ST;
                            end else state <= S_ERR;
                        end
                        OP_RUN_BURST_EXPECT: begin anum <= 6'd2;  state <= S_ARGS; end
                        default:             begin txsr <= {72'd0, UNKNOWN}; txn <= 4'd1; out_next <= S_IDLE; state <= S_OUT; end
                    endcase
                end

                S_ARGS: if (have) begin
                    rptr <= rptr + 9'd1;
                    args[acnt] <= rxb;
                    acnt <= acnt + 6'd1;
                    if (acnt + 6'd1 == anum) state <= S_EXEC;
                end

                S_EXEC: begin
                    case (op)
                        OP_SET_N: begin
                            if (args[0] >= 8'd1 && args[0] <= 8'd14) begin
                                n <= args[0][3:0];
                                state <= S_ACK;
                            end else state <= S_ERR;
                        end
                        OP_RESET: begin
                            j <= 13'd0;
                            state <= S_RST;
                        end
                        OP_LOAD: begin
                            ld_ok  <= a_fits;
                            k      <= a_addr[14:0];
                            remain <= a_cnt;
                            ldb    <= 4'd0;
                            if (a_cnt == 16'd0) state <= a_fits ? S_ACK : S_ERR;
                            else                state <= S_LDB;
                        end
                        OP_READ: begin
                            k      <= a_addr[14:0];
                            remain <= a_cnt;
                            if (!a_fits)             state <= S_ERR;
                            else if (a_cnt == 16'd0) state <= S_ACK;
                            else                     state <= S_RD;
                        end
                        OP_GATE: begin
                            gate_start <= g_ok;
                            state <= g_ok ? S_GST : S_ERR;
                        end
                        OP_EXPECT: begin
                            if (z_mask16 < {1'b0, dim}) begin
                                ez_mask <= z_mask16[13:0];
                                ez_acc  <= 80'sd0;
                                k       <= 15'd0;
                                state   <= S_EZ;
                            end else state <= S_ERR;
                        end
                        OP_LOAD_CIRCUIT: begin
                            if (circ_cnt <= 16'd128) begin
                                cm_wptr <= 8'd0;
                                cm_len  <= circ_cnt[7:0];
                                remain  <= circ_cnt;
                                gb_idx  <= 6'd0;
                                if (circ_cnt == 16'd0) state <= S_ACK;
                                else                   state <= S_LC_BYTE;
                            end else state <= S_ERR;
                        end
                        OP_RUN_BURST_EXPECT: begin
                            if (cm_len > 8'd0 && z_mask16 < {1'b0, dim}) begin
                                cm_rptr <= 8'd0;
                                burst_mode <= 1'b1;
                                expect_after_burst <= 1'b1;
                                burst_mask <= z_mask16;
                                state <= S_BST_ST;
                            end else state <= S_ERR;
                        end
                        default: state <= S_IDLE;
                    endcase
                end

                // ---- reply sequencer -------------------------------------------------
                S_OUT: begin
                    if (txn == 4'd0) state <= out_next;
                    else if (!tx_busy && !tx_start) begin
                        tx_data  <= txsr[7:0];
                        tx_start <= 1'b1;
                        txsr     <= {8'd0, txsr[79:8]};
                        txn      <= txn - 4'd1;
                        state    <= S_OUTW;
                    end
                end
                S_OUTW: state <= S_OUT;
                S_ACK: begin txsr <= {72'd0, ACK}; txn <= 4'd1; out_next <= S_IDLE; state <= S_OUT; end
                S_ERR: begin txsr <= {72'd0, ERR}; txn <= 4'd1; out_next <= S_IDLE; state <= S_OUT; end

                // ---- RESET: zero the active banks, then amplitude 0 = {0, one} ----------
                S_RST: begin
                    h_we0 <= 1'b1;
                    h_we1 <= 1'b1;
                    h_wa0 <= j;
                    h_wa1 <= j;
                    h_wd0 <= (j == 13'd0) ? {32'd0, a_one} : 64'd0;
                    h_wd1 <= 64'd0;
                    if (j == half_m1) state <= S_ACK;
                    else              j <= j + 13'd1;
                end

                // ---- LOAD: eight bytes per amplitude, re then im -----------------------
                S_LDB: if (have) begin
                    rptr <= rptr + 9'd1;
                    amp  <= {rxb, amp[63:8]};
                    ldb  <= ldb + 4'd1;
                    if (ldb == 4'd7) state <= S_LDW;
                end
                S_LDW: begin
                    if (ld_ok) begin
                        h_wa0 <= k[13:1];
                        h_wa1 <= k[13:1];
                        h_wd0 <= amp;
                        h_wd1 <= amp;
                        h_we0 <= !kpar;
                        h_we1 <= kpar;
                    end
                    k      <= k + 15'd1;
                    remain <= remain - 16'd1;
                    ldb    <= 4'd0;
                    if (remain == 16'd1) state <= ld_ok ? S_ACK : S_ERR;
                    else                 state <= S_LDB;
                end

                // ---- READ ------------------------------------------------------------
                S_RD:  begin h_ra <= k[13:1]; state <= S_RD1; end
                S_RD1: state <= S_RD2;
                S_RD2: begin
                    txsr     <= {16'd0, k_amp};
                    txn      <= 4'd8;
                    k        <= k + 15'd1;
                    remain   <= remain - 16'd1;
                    out_next <= (remain == 16'd1) ? S_ACK : S_RD;
                    state    <= S_OUT;
                end

                // ---- GATE ------------------------------------------------------------
                S_GST:   state <= S_GWAIT;
                S_GWAIT: if (!g_busy) state <= S_ACK;

                // ---- EXPECT_Z: sum of (-1)^parity(k & mask) * |amp_k|^2 -----------------
                S_EZ:  begin h_ra <= k[13:1]; state <= S_EZ1; end
                S_EZ1: state <= S_EZ2;
                S_EZ2: begin
                    ez_re <= k_amp[31:0];
                    ez_im <= k_amp[63:32];
                    state <= S_EZ3;
                end
                S_EZ3: begin
                    ez_pr <= ez_re * ez_re;
                    ez_pi <= ez_im * ez_im;
                    state <= S_EZ4;
                end
                S_EZ4: begin
                    if (^(k[13:0] & ez_mask)) ez_acc <= ez_acc - $signed({16'd0, ez_pr} + {16'd0, ez_pi});
                    else                      ez_acc <= ez_acc + $signed({16'd0, ez_pr} + {16'd0, ez_pi});
                    k <= k + 15'd1;
                    state <= (k + 15'd1 == dim) ? S_EZW : S_EZ;
                end
                S_EZW: begin
                    txsr     <= ez_acc;
                    txn      <= 4'd10;
                    out_next <= S_ACK;
                    state    <= S_OUT;
                end

                // ---- CIRCUIT LOAD (OP_LOAD_CIRCUIT): 35 bytes per gate -----------------
                S_LC_BYTE: if (have) begin
                    rptr <= rptr + 9'd1;
                    g_buf[gb_idx] <= rxb;
                    gb_idx <= gb_idx + 6'd1;
                    if (gb_idx == 6'd34) begin
                        cm_target[cm_wptr] <= g_buf[0];
                        cm_ctrl[cm_wptr]   <= {g_buf[2], g_buf[1]};
                        cm_m00r[cm_wptr]   <= {g_buf[6],  g_buf[5],  g_buf[4],  g_buf[3]};
                        cm_m00i[cm_wptr]   <= {g_buf[10], g_buf[9],  g_buf[8],  g_buf[7]};
                        cm_m01r[cm_wptr]   <= {g_buf[14], g_buf[13], g_buf[12], g_buf[11]};
                        cm_m01i[cm_wptr]   <= {g_buf[18], g_buf[17], g_buf[16], g_buf[15]};
                        cm_m10r[cm_wptr]   <= {g_buf[22], g_buf[21], g_buf[20], g_buf[19]};
                        cm_m10i[cm_wptr]   <= {g_buf[26], g_buf[25], g_buf[24], g_buf[23]};
                        cm_m11r[cm_wptr]   <= {g_buf[30], g_buf[29], g_buf[28], g_buf[27]};
                        cm_m11i[cm_wptr]   <= {rxb,       g_buf[33], g_buf[32], g_buf[31]};
                        cm_wptr            <= cm_wptr + 8'd1;
                        remain             <= remain - 16'd1;
                        gb_idx             <= 6'd0;
                        if (remain == 16'd1) state <= S_ACK;
                        else                 state <= S_LC_BYTE;
                    end
                end

                // ---- BURST EXECUTION (OP_RUN_BURST / OP_RUN_BURST_EXPECT) --------------
                S_BST_ST: begin
                    if (cur_g_ok) begin
                        gate_start <= 1'b1;
                        state      <= S_BST_WAIT;
                    end else begin
                        burst_mode <= 1'b0;
                        state      <= S_ERR;
                    end
                end
                S_BST_WAIT: if (!g_busy && !gate_start) begin
                    if (cm_rptr + 8'd1 < cm_len) begin
                        cm_rptr <= cm_rptr + 8'd1;
                        state   <= S_BST_ST;
                    end else begin
                        burst_mode <= 1'b0;
                        if (expect_after_burst) begin
                            ez_mask <= burst_mask[13:0];
                            ez_acc  <= 80'sd0;
                            k       <= 15'd0;
                            state   <= S_EZ;
                        end else begin
                            state   <= S_ACK;
                        end
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

    assign led[15]   = heartbeat_cnt[25];
    assign led[14]   = (state != S_IDLE);
    assign led[13]   = overflow;
    assign led[12]   = 1'b1;
    assign led[11:0] = 12'd1000;

endmodule
