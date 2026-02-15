#!/bin/bash
set -e

echo "[Gamesa Builder] Starting Build Process (Isolated Environment)..."

# Clean previous builds
echo "[Gamesa Builder] Cleaning..."
rm -rf dist/ build/ deb_dist/ gamesa_cortex_v2.egg-info/
mkdir -p dist/gamesa-cortex-v2/opt/gamesa-cortex-v2/gamesa_cortex_v2
mkdir -p dist/gamesa-cortex-v2/DEBIAN
mkdir -p dist/gamesa-cortex-v2/usr/bin
mkdir -p dist/gamesa-cortex-v2/lib/systemd/system

# 1. Copy Application Code to /opt/gamesa-cortex-v2/gamesa_cortex_v2
echo "[Gamesa Builder] Copying Source Code..."
cp -r gamesa_cortex_v2/src dist/gamesa-cortex-v2/opt/gamesa-cortex-v2/gamesa_cortex_v2/
cp gamesa_cortex_v2/__init__.py dist/gamesa-cortex-v2/opt/gamesa-cortex-v2/gamesa_cortex_v2/

# 2. Copy Rust Planner
mkdir -p dist/gamesa-cortex-v2/opt/gamesa-cortex-v2/gamesa_cortex_v2/rust_planner
cp -r gamesa_cortex_v2/rust_planner/src dist/gamesa-cortex-v2/opt/gamesa-cortex-v2/gamesa_cortex_v2/rust_planner/
cp gamesa_cortex_v2/rust_planner/Cargo.toml dist/gamesa-cortex-v2/opt/gamesa-cortex-v2/gamesa_cortex_v2/rust_planner/

# 3. Copy Dashboard
echo "[Gamesa Builder] Copying Dashboard..."
mkdir -p dist/gamesa-cortex-v2/opt/gamesa-cortex-v2/gamesa_cortex_v2/dashboard
cp -r gamesa_cortex_v2/dashboard/* dist/gamesa-cortex-v2/opt/gamesa-cortex-v2/gamesa_cortex_v2/dashboard/

# 4. Create Launchers (Using VENV Python)
echo "[Gamesa Builder] Creating Launchers..."

# Dashboard Launcher
cat <<EOF > dist/gamesa-cortex-v2/usr/bin/gamesa-dashboard
#!/bin/bash
export PYTHONPATH=/opt/gamesa-cortex-v2
/opt/gamesa-cortex-v2/venv/bin/python -m streamlit run /opt/gamesa-cortex-v2/gamesa_cortex_v2/dashboard/app.py
EOF
chmod +x dist/gamesa-cortex-v2/usr/bin/gamesa-dashboard

# API Launcher
cat <<EOF > dist/gamesa-cortex-v2/usr/bin/gamesa-api
#!/bin/bash
export PYTHONPATH=/opt/gamesa-cortex-v2
/opt/gamesa-cortex-v2/venv/bin/python -m uvicorn gamesa_cortex_v2.src.core.api:app --host 0.0.0.0 --port 8000
EOF
chmod +x dist/gamesa-cortex-v2/usr/bin/gamesa-api

# 5. Create Systemd Service
cat <<EOF > dist/gamesa-cortex-v2/lib/systemd/system/gamesa-api.service
[Unit]
Description=Gamesa Cortex V2 API Service
After=network.target

[Service]
Type=simple
User=root
ExecStart=/usr/bin/gamesa-api
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF

# 6. Create Control File
echo "[Gamesa Builder] Creating Control File..."
cat <<EOF > dist/gamesa-cortex-v2/DEBIAN/control
Package: gamesa-cortex-v2
Version: 0.1.0
Section: python
Priority: optional
Architecture: all
Depends: python3, python3-venv, python3-pip
Maintainer: Gamesa Cortex Team <dev@gamesacortex.com>
Description: The Neural Control Plane for Industry 5.0
 Gamesa Cortex V2 orchestrates AI inference, economic planning, and safety checks.
 This package installs into /opt/gamesa-cortex-v2 and creates a private virtual environment.
EOF

# 7. Create Post-Install Script (The Magic)
echo "[Gamesa Builder] Creating Post-Install Script..."
cat <<EOF > dist/gamesa-cortex-v2/DEBIAN/postinst
#!/bin/bash
set -e

if [ "\$1" = "configure" ]; then
    echo "[Gamesa Setup] Creating Virtual Environment in /opt/gamesa-cortex-v2/venv..."
    if [ ! -d "/opt/gamesa-cortex-v2/venv" ]; then
        python3 -m venv /opt/gamesa-cortex-v2/venv
    fi

    echo "[Gamesa Setup] Installing Dependencies into VENV..."
    source /opt/gamesa-cortex-v2/venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install core dependencies
    # Note: openvino might take time or fail on non-x86, so we utilize || true for robustness in this demo
    pip install numpy psutil pandas altair streamlit fastapi uvicorn
    
    echo "[Gamesa Setup] Installation Complete."
fi

exit 0
EOF
chmod 755 dist/gamesa-cortex-v2/DEBIAN/postinst

# 8. Build .deb
echo "[Gamesa Builder] Building .deb package..."
dpkg-deb --build dist/gamesa-cortex-v2 dist/gamesa-cortex-v2_0.1.0_all.deb

echo "[Gamesa Builder] Build Complete: dist/gamesa-cortex-v2_0.1.0_all.deb"

