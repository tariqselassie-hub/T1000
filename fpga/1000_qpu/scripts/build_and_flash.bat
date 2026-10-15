@echo off
setlocal
echo ======================================================================
echo  1000_QPU - ARTIX-7 XC7A35T BUILD & FLASH AUTOMATION
echo ======================================================================

set VIVADO_BIN=C:\AMDDesignTools\2026.1\Vivado\bin\vivado.bat

if not exist "%VIVADO_BIN%" (
    echo [ERROR] Vivado not found at %VIVADO_BIN%
    pause
    exit /b 1
)

echo [STEP 1/3] Generating Vivado Project: 1000_qpu.xpr ...
"%VIVADO_BIN%" -mode batch -nolog -nojournal -source "%~dp0create_project.tcl"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Project generation failed!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [STEP 2/3] Synthesizing, Routing and Generating Bitstream (build/1000_qpu.bit) ...
"%VIVADO_BIN%" -mode batch -nolog -nojournal -source "%~dp0build_bitstream.tcl"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Bitstream generation failed!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [STEP 3/3] Programming Artix-7 XC7A35T Hardware over JTAG ...
"%VIVADO_BIN%" -mode batch -nolog -nojournal -source "%~dp0flash_bitstream.tcl"
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] JTAG auto-programming encountered an issue.
    echo Ensure the board is plugged in via USB and powered ON.
    echo You can also open 1000_qpu.xpr in Vivado to program the hardware manually.
)

echo.
echo ======================================================================
echo  [DONE] 1000_QPU Project Ready!
echo  Vivado Project Location: %~dp0..\1000_qpu.xpr
echo ======================================================================
endlocal
