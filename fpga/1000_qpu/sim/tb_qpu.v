// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

`timescale 1ns / 1ps

// Replays vectors/stim.mem into qpu_top over a 3 Mbaud serial line and compares every byte the
// design sends back with vectors/expect.mem (the reply of the Rust model). Real baud rate is
// kept so the design sees the same command spacing as on the board.
module tb_qpu;

    localparam CLK_PERIOD = 10;
    localparam BIT_CLKS   = 33;           // 100 MHz / 3 Mbaud, as in uart_rx/uart_tx
    localparam MAXB       = 262144;

    reg clk = 1'b0;
    reg rst = 1'b1;
    reg rx_line = 1'b1;
    wire tx_line;
    wire [15:0] led;
    wire [6:0]  seg;
    wire        dp;
    wire [3:0]  an;

    always #(CLK_PERIOD / 2) clk = ~clk;

    qpu_top dut (
        .clk(clk), .rst(rst), .led(led), .seg(seg), .dp(dp), .an(an),
        .uart_tx(tx_line), .uart_rx(rx_line)
    );

    reg [7:0]  stim   [0:MAXB-1];
    reg [7:0]  expect [0:MAXB-1];
    reg [31:0] sizes  [0:2];
    reg [31:0] ftab   [0:65535];
    integer n_frames, f, fl, rl, fb, reply_due;
    integer stim_len, exp_len;
    integer sent = 0;
    integer got = 0;
    integer errors = 0;

    // ---- transmitter: host -> design ---------------------------------------------
    task send_byte(input [7:0] b);
        integer i;
        begin
            rx_line = 1'b0;
            repeat (BIT_CLKS) @(posedge clk);
            for (i = 0; i < 8; i = i + 1) begin
                rx_line = b[i];
                repeat (BIT_CLKS) @(posedge clk);
            end
            rx_line = 1'b1;
            repeat (BIT_CLKS) @(posedge clk);
        end
    endtask

    // ---- receiver: design -> host --------------------------------------------------
    reg [7:0] rb;
    integer ri;
    initial begin : monitor
        forever begin
            @(negedge tx_line);
            repeat (BIT_CLKS + BIT_CLKS / 2) @(posedge clk);
            for (ri = 0; ri < 8; ri = ri + 1) begin
                rb[ri] = tx_line;
                if (ri != 7) repeat (BIT_CLKS) @(posedge clk);
            end
            if (got < exp_len) begin
                if (rb !== expect[got]) begin
                    errors = errors + 1;
                    if (errors <= 12)
                        $display("MISMATCH reply byte %0d: got %02x expected %02x (t=%0t)", got, rb, expect[got], $time);
                end
            end else begin
                errors = errors + 1;
                $display("EXTRA reply byte %0d: %02x", got, rb);
            end
            got = got + 1;
            repeat (BIT_CLKS / 2 + 1) @(posedge clk);
        end
    end

    // ---- sequence ------------------------------------------------------------------
    integer idle;
    initial begin
        $readmemh("vectors/sizes.mem", sizes);
        $readmemh("vectors/stim.mem", stim);
        $readmemh("vectors/expect.mem", expect);
        stim_len = sizes[0];
        exp_len  = sizes[1];
        n_frames = sizes[2];
        $readmemh("vectors/frames.mem", ftab);
        repeat (20) @(posedge clk);
        rst = 1'b0;
        repeat (20) @(posedge clk);
        // Frames with a reply longer than one byte are sent stop-and-wait (the device has a
        // 512-byte receive FIFO and answers while the next bytes arrive); one-byte replies
        // (gates, loads, errors) are streamed back to back.
        reply_due = 0;
        for (f = 0; f < n_frames; f = f + 1) begin
            fl = ftab[2 * f];
            rl = ftab[2 * f + 1];
            for (fb = 0; fb < fl; fb = fb + 1) begin
                send_byte(stim[sent]);
                sent = sent + 1;
            end
            reply_due = reply_due + rl;
            if (rl > 1) begin
                idle = 0;
                while (got < reply_due && idle < 2000000) begin
                    @(posedge clk);
                    idle = idle + 1;
                end
            end
        end
        // allow the last replies to drain
        idle = 0;
        while (got < exp_len && idle < 400000) begin
            @(posedge clk);
            idle = idle + 1;
        end
        repeat (4000) @(posedge clk);
        $display("sent %0d bytes, received %0d of %0d expected reply bytes, %0d errors",
                 sent, got, exp_len, errors);
        if (errors == 0 && got == exp_len) $display("RESULT: PASS");
        else                               $display("RESULT: FAIL");
        $finish;
    end

endmodule
