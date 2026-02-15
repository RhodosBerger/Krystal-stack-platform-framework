#!/bin/bash
set -e

echo "[Gamesa Builder] Starting Build Process..."

# Clean previous builds
echo "[Gamesa Builder] Cleaning..."
rm -rf dist/ build/ deb_dist/ gamesa_cortex_v2.egg-info/
mkdir -p dist/gamesa-cortex-v2/DEBIAN
mkdir -p dist/gamesa-cortex-v2/usr/lib/python3/dist-packages/gamesa_cortex_v2

# 1. Copy Python Source Code
echo "[Gamesa Builder] Copying Source Code..."
cp -r gamesa_cortex_v2/src/* dist/gamesa-cortex-v2/usr/lib/python3/dist-packages/gamesa_cortex_v2/
cp gamesa_cortex_v2/__init__.py dist/gamesa-cortex-v2/usr/lib/python3/dist-packages/gamesa_cortex_v2/

# 2. Copy Rust Planner (Source only, since we can't compile)
# In a real build, we'd put the .so file here.
mkdir -p dist/gamesa-cortex-v2/usr/lib/python3/dist-packages/gamesa_cortex_v2/rust_planner
cp -r gamesa_cortex_v2/rust_planner/src dist/gamesa-cortex-v2/usr/lib/python3/dist-packages/gamesa_cortex_v2/rust_planner/
cp gamesa_cortex_v2/rust_planner/Cargo.toml dist/gamesa-cortex-v2/usr/lib/python3/dist-packages/gamesa_cortex_v2/rust_planner/

# 2.5 Copy Dashboard
echo "[Gamesa Builder] Copying Dashboard..."
mkdir -p dist/gamesa-cortex-v2/usr/lib/python3/dist-packages/gamesa_cortex_v2/dashboard
cp -r gamesa_cortex_v2/dashboard/* dist/gamesa-cortex-v2/usr/lib/python3/dist-packages/gamesa_cortex_v2/dashboard/

# 2.6 Create Dashboard Launcher
mkdir -p dist/gamesa-cortex-v2/usr/bin
cat <<EOF > dist/gamesa-cortex-v2/usr/bin/gamesa-dashboard
#!/bin/bash
export PYTHONPATH=/usr/lib/python3/dist-packages:\$PYTHONPATH
streamlit run /usr/lib/python3/dist-packages/gamesa_cortex_v2/dashboard/app.py
EOF
chmod +x dist/gamesa-cortex-v2/usr/bin/gamesa-dashboard

# 3. Create Control File
echo "[Gamesa Builder] Creating Control File..."
cat <<EOF > dist/gamesa-cortex-v2/DEBIAN/control
Package: gamesa-cortex-v2
Version: 0.1.0
Section: python
Priority: optional
Architecture: all
Depends: python3, python3-numpy, python3-psutil
Maintainer: Gamesa Cortex Team <dev@gamesacortex.com>
Description: The Neural Control Plane for Industry 5.0
 Gamesa Cortex V2 orchestrates AI inference, economic planning, and safety checks.
EOF

# 4. Build .deb
echo "[Gamesa Builder] Building .deb package..."
dpkg-deb --build dist/gamesa-cortex-v2 dist/gamesa-cortex-v2_0.1.0_all.deb

echo "[Gamesa Builder] Build Complete: dist/gamesa-cortex-v2_0.1.0_all.deb"
