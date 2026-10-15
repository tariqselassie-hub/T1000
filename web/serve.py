# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""
Local Web Server & Launcher for T-1000 3D Quantum Molecular Dashboard.
Serves the web dashboard locally on an available port and optionally launches the browser.
"""

import http.server
import socketserver
import webbrowser
from pathlib import Path
import sys

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PORT = 8000
WEB_DIR = Path(__file__).resolve().parent

class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def log_message(self, format, *args):
        # Mute normal GET requests for clean terminal output
        if args and str(args[1]) not in ("200", "304"):
            super().log_message(format, *args)

def main():
    # Allow port reuse to avoid 'Address already in use' errors
    socketserver.TCPServer.allow_reuse_address = True
    
    with socketserver.TCPServer(("", PORT), QuietHandler) as httpd:
        url = f"http://localhost:{PORT}"
        print("╔═══════════════════════════════════════════════════════════════════════════════╗")
        print("║        T-1000 3D QUANTUM MOLECULAR & ATOMIC MESH DASHBOARD SERVER             ║")
        print("╚═══════════════════════════════════════════════════════════════════════════════╝")
        print(f" [URL ENDPOINT]  : {url}")
        print(f" [ASSETS DIR]    : {WEB_DIR}")
        print(" [STATUS]        : ACTIVE (Press Ctrl+C to terminate)")
        print(" [SILICON LINK]  : AMD Artix-7 Basys 3 (COM6 @ 3.0 MBaud)")
        print("═════════════════════════════════════════════════════════════════════════════════\n")
        
        # Open in default system browser if run directly
        if "--no-browser" not in sys.argv:
            webbrowser.open(url)
            
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[-] Shutting down T-1000 Web Dashboard Server.")

if __name__ == "__main__":
    main()
