#!/usr/bin/env python3
"""
Krystal-Stack Localhost Production Launcher & Watchdog
======================================================
Launches the Localhost Hub Server and opens the Mission Control in the default browser.
"""

import sys
import os
import time
import socket
import webbrowser
import subprocess

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def find_available_port(start_port=8080, max_attempts=5):
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('127.0.0.1', port)) != 0:
                return port
    return start_port

def main():
    port = find_available_port(8080)
    url = f"http://localhost:{port}/"

    print("\n" + "="*70)
    print(" [KRYSTAL] KRYSTAL-STACK LOCALHOST PRODUCTION LAUNCHER")
    print("="*70)
    print(f" Target Port: {port}")
    print(f" Launch URL:  {url}")
    print("="*70 + "\n")

    # Import and start the server
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    try:
        from krystal_web_hub.server import start_server
        
        # Schedule browser launch 1 second after server starts
        def open_browser():
            time.sleep(1.0)
            print(f"[LAUNCHER] Opening Mission Control in browser: {url}")
            webbrowser.open(url)

        import threading
        threading.Thread(target=open_browser, daemon=True).start()

        start_server(host="127.0.0.1", port=port)

    except Exception as e:
        print(f"[ERROR] Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
