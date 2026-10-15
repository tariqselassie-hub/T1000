# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

# =============================================================================
# Project: 1000_QPU - Vivado Project Creation Script
# Target:  Xilinx Artix-7 XC7A35T-1CPG236C (Digilent Basys 3)
# =============================================================================

set SCRIPT_DIR [file dirname [file normalize [info script]]]
set PROJ_DIR   [file dirname $SCRIPT_DIR]

puts "======================================================================"
puts ">> \[1000_QPU\]: Creating Vivado Project in: $PROJ_DIR"
puts "======================================================================"

# Create 1000_qpu project targeting Artix-7 XC7A35T
create_project 1000_qpu $PROJ_DIR -part xc7a35tcpg236-1 -force

# Set project properties
set_property target_language Verilog [current_project]
set_property default_lib xil_defaultlib [current_project]

# Add Verilog Design Sources
add_files -fileset sources_1 [glob $PROJ_DIR/rtl/*.v]

# Add Timing and Pin Constraints
add_files -fileset constrs_1 $PROJ_DIR/constraints/basys3_1000_qpu.xdc

# Set Top Module
set_property top qpu_top [current_fileset]
update_compile_order -fileset sources_1

puts "======================================================================"
puts ">> \[1000_QPU\]: Project Created Successfully: $PROJ_DIR/1000_qpu.xpr"
puts "======================================================================"
