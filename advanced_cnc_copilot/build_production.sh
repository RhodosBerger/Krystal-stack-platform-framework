#!/bin/bash
set -e

# Version
VERSION="1.4.0"
PKG_NAME="advanced-cnc-copilot-production"
ARCH="amd64"
DEB_NAME="${PKG_NAME}_${VERSION}_${ARCH}.deb"
PKG_DIR="build/${PKG_NAME}_${VERSION}_${ARCH}"

echo "Building Production Package: ${DEB_NAME}..."

# 1. Cleanup
rm -rf build/${PKG_NAME}_${VERSION}_${ARCH}
mkdir -p ${PKG_DIR}/opt/advanced_cnc_copilot
mkdir -p ${PKG_DIR}/DEBIAN

# 2. Copy Codebase (Core Only)
echo "Copying Backend Files..."
cp -r backend ${PKG_DIR}/opt/advanced_cnc_copilot/
cp main.py ${PKG_DIR}/opt/advanced_cnc_copilot/


# Handle Requirements
if [ -f "requirements.txt" ]; then
    cp requirements.txt ${PKG_DIR}/opt/advanced_cnc_copilot/
elif [ -f "../requirements.txt" ]; then
    cp ../requirements.txt ${PKG_DIR}/opt/advanced_cnc_copilot/
else
    touch ${PKG_DIR}/opt/advanced_cnc_copilot/requirements.txt
fi

# Copy Models (Production needs the model too)
if [ -f "../phantom_net.onnx" ]; then
    cp ../phantom_net.onnx ${PKG_DIR}/opt/advanced_cnc_copilot/backend/cms/services/
fi

# 3. Create Control File
cat > ${PKG_DIR}/DEBIAN/control <<EOF
Package: ${PKG_NAME}
Version: ${VERSION}
Section: science
Priority: optional
Architecture: ${ARCH}
Maintainer: Dusan <dusan@example.com>
Description: Advanced CNC Copilot (Cortex Engine) - Production Release
  Optimized for deployment. Includes Parallel Streamer and OpenVINO Engine.
EOF

# 4. Create Post-Install Script
cat > ${PKG_DIR}/DEBIAN/postinst <<EOF
#!/bin/bash
echo "Installing Production Dependencies..."
pip3 install -r /opt/advanced_cnc_copilot/requirements.txt
echo "Cortex Engine (Production) Installed."
EOF
chmod 755 ${PKG_DIR}/DEBIAN/postinst

# 5. Build Deb
mkdir -p dist
dpkg-deb --build ${PKG_DIR}
mv build/${DEB_NAME} dist/

echo "Production Build Complete: dist/${DEB_NAME}"
