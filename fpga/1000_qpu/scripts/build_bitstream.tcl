# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

# =============================================================================
# Project: 1000_QPU - Vivado Non-Project Batch Synthesis & Implementation Script
# Target:  Xilinx Artix-7 XC7A35T-1CPG236C (Digilent Basys 3)
# Output:  build/1000_qpu.bit
# =============================================================================

set SCRIPT_DIR [file dirname [file normalize [info script]]]
set PROJ_DIR   [file dirname $SCRIPT_DIR]
set BUILD_DIR  "$PROJ_DIR/build"

file mkdir $BUILD_DIR

puts "======================================================================"
puts ">> \[1000_QPU\]: Starting Hardware Synthesis for Artix-7 XC7A35T..."
puts "======================================================================"

# Read design sources
read_verilog [glob $PROJ_DIR/rtl/*.v]
read_xdc $PROJ_DIR/constraints/basys3_1000_qpu.xdc

# Run Synthesis
synth_design -top qpu_top -part xc7a35tcpg236-1 -flatten_hierarchy rebuilt
write_checkpoint -force $BUILD_DIR/post_synth.dcp
report_utilization -file $BUILD_DIR/utilization_synth.rpt

puts "======================================================================"
puts ">> \[1000_QPU\]: Synthesis Complete. Running Logic Optimization..."
puts "======================================================================"

opt_design
write_checkpoint -force $BUILD_DIR/post_opt.dcp

puts "======================================================================"
puts ">> \[1000_QPU\]: Running Placement and Timing Optimization..."
puts "======================================================================"

place_design
write_checkpoint -force $BUILD_DIR/post_place.dcp

puts "======================================================================"
puts ">> \[1000_QPU\]: Running Routing and Timing Analysis..."
puts "======================================================================"

route_design
write_checkpoint -force $BUILD_DIR/post_route.dcp
report_timing_summary -file $BUILD_DIR/timing_summary.rpt
report_utilization -file $BUILD_DIR/utilization_route.rpt

puts "======================================================================"
puts ">> \[1000_QPU\]: Generating Physical Bitstream: $BUILD_DIR/1000_qpu.bit"
puts "======================================================================"

write_bitstream -force -bin_file $BUILD_DIR/1000_qpu.bit

puts "======================================================================"
puts ">> \[1000_QPU\]: BITSTREAM GENERATION COMPLETE!"
puts ">> Target: $BUILD_DIR/1000_qpu.bit"
puts "======================================================================"
