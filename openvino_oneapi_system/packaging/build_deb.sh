#!/usr/bin/env sh
set -eu

PKG_NAME="openvino-oneapi-system"
VERSION="${1:-1.0.0}"
ARCH="$(dpkg --print-architecture)"
ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
BUILD_DIR="$ROOT_DIR/packaging/.build/${PKG_NAME}_${VERSION}_${ARCH}"
DIST_DIR="$ROOT_DIR/dist"
INSTALL_DIR="/usr/lib/$PKG_NAME"
SERVICE_NAME="$PKG_NAME.service"
SERVICE_USER="openvino-ovo"

rm -rf "$BUILD_DIR"
mkdir -p \
  "$BUILD_DIR/DEBIAN" \
  "$BUILD_DIR$INSTALL_DIR" \
  "$BUILD_DIR/usr/bin" \
  "$BUILD_DIR/etc/default" \
  "$BUILD_DIR/lib/systemd/system" \
  "$BUILD_DIR/var/log/$PKG_NAME" \
  "$BUILD_DIR/var/lib/$PKG_NAME" \
  "$DIST_DIR"

# Package control metadata
cat > "$BUILD_DIR/DEBIAN/control" <<EOF
Package: $PKG_NAME
Version: $VERSION
Section: utils
Priority: optional
Architecture: $ARCH
Maintainer: Dušan Kopecký <dusan.kopecky0101@gmail.com>
Depends: python3 (>= 3.10)
Recommends: sysbench
Description: OpenVINO + oneAPI orchestration and Linux benchmark toolkit
 Includes adaptive planning, delegated logging, 3D grid memory abstraction,
 algorithmic planning, and synthetic+sysbench benchmark workflow.
 Licensed under Apache 2.0.
EOF

# Copy application payload, excluding runtime logs and cache artifacts.
cp "$ROOT_DIR/main.py" "$BUILD_DIR$INSTALL_DIR/main.py"
cp "$ROOT_DIR/benchmark_linux.py" "$BUILD_DIR$INSTALL_DIR/benchmark_linux.py"
cp "$ROOT_DIR/README.md" "$BUILD_DIR$INSTALL_DIR/README.md"
cp "$ROOT_DIR/requirements.txt" "$BUILD_DIR$INSTALL_DIR/requirements.txt"
cp -r "$ROOT_DIR/ovo" "$BUILD_DIR$INSTALL_DIR/ovo"
[ -f "$ROOT_DIR/LICENSE" ] && cp "$ROOT_DIR/LICENSE" "$BUILD_DIR$INSTALL_DIR/LICENSE"
[ -f "$ROOT_DIR/phantom_net.onnx" ] && cp "$ROOT_DIR/phantom_net.onnx" "$BUILD_DIR$INSTALL_DIR/phantom_net.onnx"

find "$BUILD_DIR$INSTALL_DIR" -type d -name "__pycache__" -prune -exec rm -rf {} +
find "$BUILD_DIR$INSTALL_DIR" -type f -name "*.pyc" -delete

# Wrapper binaries.
cat > "$BUILD_DIR/usr/bin/ovo-runtime" <<'EOF'
#!/usr/bin/env sh
set -eu
[ -f /etc/default/openvino-oneapi-system ] && . /etc/default/openvino-oneapi-system
export OVO_LOG_DIR="${OVO_LOG_DIR:-/var/log/openvino-oneapi-system}"
exec python3 /usr/lib/openvino-oneapi-system/main.py "$@"
EOF

cat > "$BUILD_DIR/usr/bin/ovo-benchmark" <<'EOF'
#!/usr/bin/env sh
set -eu
[ -f /etc/default/openvino-oneapi-system ] && . /etc/default/openvino-oneapi-system
export OVO_LOG_DIR="${OVO_LOG_DIR:-/var/log/openvino-oneapi-system}"
exec python3 /usr/lib/openvino-oneapi-system/benchmark_linux.py "$@"
EOF

chmod 0755 "$BUILD_DIR/usr/bin/ovo-runtime" "$BUILD_DIR/usr/bin/ovo-benchmark"

# Default runtime environment (conffile).
cat > "$BUILD_DIR/etc/default/openvino-oneapi-system" <<'EOF'
# openvino-oneapi-system default environment
OVO_LOG_DIR=/var/log/openvino-oneapi-system
# For service mode, cycles=0 means run forever.
OVO_RUNTIME_ARGS="--cycles 0 --interval 0.5"
EOF

# Optional systemd service.
cat > "$BUILD_DIR/lib/systemd/system/$SERVICE_NAME" <<'EOF'
[Unit]
Description=OpenVINO ONE API Runtime
After=network.target

[Service]
Type=simple
User=openvino-ovo
Group=openvino-ovo
EnvironmentFile=-/etc/default/openvino-oneapi-system
WorkingDirectory=/var/lib/openvino-oneapi-system
ExecStart=/bin/sh -lc 'exec /usr/bin/ovo-runtime ${OVO_RUNTIME_ARGS:-"--cycles 0 --interval 0.5"}'
Restart=on-failure
RestartSec=2

[Install]
WantedBy=multi-user.target
EOF

# Mark config file as conffile.
cat > "$BUILD_DIR/DEBIAN/conffiles" <<'EOF'
/etc/default/openvino-oneapi-system
EOF

# Lifecycle scripts.
cat > "$BUILD_DIR/DEBIAN/postinst" <<'EOF'
#!/usr/bin/env sh
set -eu
if ! getent group openvino-ovo >/dev/null 2>&1; then
    addgroup --system openvino-ovo >/dev/null 2>&1 || true
fi
if ! getent passwd openvino-ovo >/dev/null 2>&1; then
    adduser --system --ingroup openvino-ovo --home /var/lib/openvino-oneapi-system \
        --no-create-home --shell /usr/sbin/nologin openvino-ovo >/dev/null 2>&1 || true
fi
mkdir -p /var/log/openvino-oneapi-system /var/lib/openvino-oneapi-system
chown -R openvino-ovo:openvino-ovo /var/log/openvino-oneapi-system /var/lib/openvino-oneapi-system || true
chmod 0775 /var/log/openvino-oneapi-system || true
if command -v systemctl >/dev/null 2>&1; then
    systemctl daemon-reload >/dev/null 2>&1 || true
fi
exit 0
EOF
chmod 0755 "$BUILD_DIR/DEBIAN/postinst"

cat > "$BUILD_DIR/DEBIAN/prerm" <<'EOF'
#!/usr/bin/env sh
set -eu
if command -v systemctl >/dev/null 2>&1; then
    systemctl stop openvino-oneapi-system.service >/dev/null 2>&1 || true
    systemctl disable openvino-oneapi-system.service >/dev/null 2>&1 || true
fi
exit 0
EOF
chmod 0755 "$BUILD_DIR/DEBIAN/prerm"

cat > "$BUILD_DIR/DEBIAN/postrm" <<'EOF'
#!/usr/bin/env sh
set -eu
if command -v systemctl >/dev/null 2>&1; then
    systemctl daemon-reload >/dev/null 2>&1 || true
fi
exit 0
EOF
chmod 0755 "$BUILD_DIR/DEBIAN/postrm"

OUT_DEB="$DIST_DIR/${PKG_NAME}_${VERSION}_${ARCH}.deb"
dpkg-deb --root-owner-group --build "$BUILD_DIR" "$OUT_DEB"
echo "Built: $OUT_DEB"
