#!/bin/bash
set -e

# Version
VERSION="1.4.0"
PKG_NAME="advanced-cnc-copilot-insider"
ARCH="amd64"
DEB_NAME="${PKG_NAME}_${VERSION}_${ARCH}.deb"
PKG_DIR="build/${PKG_NAME}_${VERSION}_${ARCH}"

echo "Building ${DEB_NAME}..."

# 1. Cleanup
rm -rf build
mkdir -p ${PKG_DIR}/opt/advanced_cnc_copilot
mkdir -p ${PKG_DIR}/DEBIAN

# 2. Copy Codebase
echo "Copying Backend Files..."
cp -r backend ${PKG_DIR}/opt/advanced_cnc_copilot/
cp main.py ${PKG_DIR}/opt/advanced_cnc_copilot/
if [ -f "requirements.txt" ]; then
    cp requirements.txt ${PKG_DIR}/opt/advanced_cnc_copilot/
elif [ -f "../requirements.txt" ]; then
    cp ../requirements.txt ${PKG_DIR}/opt/advanced_cnc_copilot/
else
    echo "WARNING: requirements.txt not found!"
    touch ${PKG_DIR}/opt/advanced_cnc_copilot/requirements.txt
fi
cp INSIDER_README.md ${PKG_DIR}/opt/advanced_cnc_copilot/README.md

# Copy Models & Scripts
if [ -f "../phantom_net.onnx" ]; then
    cp ../phantom_net.onnx ${PKG_DIR}/opt/advanced_cnc_copilot/backend/cms/services/
    echo "Included Neural Model."
else
    echo "WARNING: phantom_net.onnx not found in root! Package will use Heuristic Fallback."
fi

chmod +x ${PKG_DIR}/opt/advanced_cnc_copilot/backend/benchmarks/stress_test_cortex.sh

# 3. Create Control File
cat > ${PKG_DIR}/DEBIAN/control <<EOF
Package: ${PKG_NAME}
Version: ${VERSION}
Section: science
Priority: optional
Architecture: ${ARCH}
Maintainer: Dusan <dusan@example.com>
Description: Advanced CNC Copilot (Cortex Engine)
  Includes Parallel Streamer, Evolutionary Optimizer, and Hex Trace Logger.
  Powered by OpenVINO and Dopamine Engine.
EOF

# 4. Create Post-Install Script (Dependencies)
cat > ${PKG_DIR}/DEBIAN/postinst <<EOF
#!/bin/bash
echo "Installing Python Dependencies..."
pip3 install -r /opt/advanced_cnc_copilot/requirements.txt
echo "Cortex Engine Installed Successfully."
EOF
chmod 755 ${PKG_DIR}/DEBIAN/postinst

# 5. Build Deb
dpkg-deb --build ${PKG_DIR}
mv build/${DEB_NAME} .

echo "Build Complete: ${DEB_NAME}"
