// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

`timescale 1ns / 1ps

//////////////////////////////////////////////////////////////////////////////////
// Module: qpu_bram_matrix
// Description: Dual True Dual-Port Block RAM Matrix for 14-Qubit Statevector
//              Infers true RAMB36E1 hardware primitives cleanly with Port A and Port B.
// Target: Xilinx Artix-7 XC7A35T
//////////////////////////////////////////////////////////////////////////////////

module qpu_bram_matrix #(
    parameter ADDR_WIDTH = 14, // 2^14 = 16,384 amplitudes
    parameter DATA_WIDTH = 32  // 16-bit Re + 16-bit Im
)(
    input  wire                  clk,
    input  wire                  bank_sel, // 0: Read Bank 0 / Write Bank 1; 1: Vice-versa
    
    // Read Ports (Port A & Port B)
    input  wire [ADDR_WIDTH-1:0] raddr_0,
    input  wire [ADDR_WIDTH-1:0] raddr_1,
    output wire [DATA_WIDTH-1:0] rdata_0,
    output wire [DATA_WIDTH-1:0] rdata_1,
    
    // Write Ports (Port A & Port B)
    input  wire                  we,
    input  wire [ADDR_WIDTH-1:0] waddr_0,
    input  wire [ADDR_WIDTH-1:0] waddr_1,
    input  wire [DATA_WIDTH-1:0] wdata_0,
    input  wire [DATA_WIDTH-1:0] wdata_1
);

    // =========================================================================
    // Bank 0: True Dual-Port Block RAM (Port A and Port B separate processes)
    // =========================================================================
    (* ram_style = "block" *) reg [DATA_WIDTH-1:0] bank0 [0:(1<<ADDR_WIDTH)-1];
    reg [DATA_WIDTH-1:0] bank0_dout_a;
    reg [DATA_WIDTH-1:0] bank0_dout_b;

    wire we_bank0 = we && (bank_sel == 1'b1);
    wire [ADDR_WIDTH-1:0] addr_b0_a = we_bank0 ? waddr_0 : raddr_0;
    wire [ADDR_WIDTH-1:0] addr_b0_b = we_bank0 ? waddr_1 : raddr_1;

    // Port A
    always @(posedge clk) begin
        if (we_bank0)
            bank0[addr_b0_a] <= wdata_0;
        bank0_dout_a <= bank0[addr_b0_a];
    end

    // Port B
    always @(posedge clk) begin
        if (we_bank0)
            bank0[addr_b0_b] <= wdata_1;
        bank0_dout_b <= bank0[addr_b0_b];
    end

    // =========================================================================
    // Bank 1: True Dual-Port Block RAM (Port A and Port B separate processes)
    // =========================================================================
    (* ram_style = "block" *) reg [DATA_WIDTH-1:0] bank1 [0:(1<<ADDR_WIDTH)-1];
    reg [DATA_WIDTH-1:0] bank1_dout_a;
    reg [DATA_WIDTH-1:0] bank1_dout_b;

    wire we_bank1 = we && (bank_sel == 1'b0);
    wire [ADDR_WIDTH-1:0] addr_b1_a = we_bank1 ? waddr_0 : raddr_0;
    wire [ADDR_WIDTH-1:0] addr_b1_b = we_bank1 ? waddr_1 : raddr_1;

    // Port A
    always @(posedge clk) begin
        if (we_bank1)
            bank1[addr_b1_a] <= wdata_0;
        bank1_dout_a <= bank1[addr_b1_a];
    end

    // Port B
    always @(posedge clk) begin
        if (we_bank1)
            bank1[addr_b1_b] <= wdata_1;
        bank1_dout_b <= bank1[addr_b1_b];
    end

    // Output multiplexer: source bank selection
    assign rdata_0 = (!bank_sel) ? bank0_dout_a : bank1_dout_a;
    assign rdata_1 = (!bank_sel) ? bank0_dout_b : bank1_dout_b;

endmodule
