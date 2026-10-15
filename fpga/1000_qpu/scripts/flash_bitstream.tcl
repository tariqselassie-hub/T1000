# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

# =============================================================================
# Project: 1000_QPU - Artix-7 JTAG Bitstream Programming Script
# Target:  xc7a35tcpg236-1 (Basys 3 on USB)
# =============================================================================

set SCRIPT_DIR [file dirname [file normalize [info script]]]
set PROJ_DIR   [file dirname $SCRIPT_DIR]
set BIT_PATH   "$PROJ_DIR/build/1000_qpu.bit"

if {![file exists $BIT_PATH]} {
    puts ">> \[ERROR\]: Bitstream not found at $BIT_PATH"
    puts ">> Please run build_bitstream.tcl first."
    exit 1
}

puts "======================================================================"
puts ">> \[1000_QPU\]: Connecting to Vivado Hardware Server (localhost:3121)..."
puts "======================================================================"

open_hw_manager
connect_hw_server -url localhost:3121 -allow_non_jtag

if {[llength [get_hw_targets]] == 0} {
    puts ">> \[ERROR\]: No JTAG hardware target detected!"
    puts ">> Ensure the Basys 3 board is connected via USB and powered ON."
    disconnect_hw_server
    close_hw_manager
    exit 1
}

set hw_target [lindex [get_hw_targets] 0]
open_hw_target $hw_target
puts ">> \[1000_QPU\]: Connected to Target: $hw_target"

set hw_device [lindex [get_hw_devices] 0]
current_hw_device $hw_device
refresh_hw_device -update_hw_probes false $hw_device

puts ">> \[1000_QPU\]: Programming Device $hw_device with $BIT_PATH..."
set_property PROGRAM.FILE $BIT_PATH $hw_device
program_hw_devices $hw_device

puts "======================================================================"
puts ">> \[1000_QPU\]: PROGRAMMING SUCCESSFUL!"
puts ">> Status: \"1000\" is now glowing on the 7-segment display!"
puts "======================================================================"

disconnect_hw_server
close_hw_manager
exit 0
