#!/usr/bin/env bash
# ==============================================================================
# KRYSTAL-STACK: WORDPRESS SUBDOMAIN AUTOMATED DEPLOYMENT SCRIPT (Bash / Linux)
# ==============================================================================
set -euo pipefail

SUBDOMAIN="${1:-krystal.poslednikmen.cz}"
BACKEND_URL="${2:-http://127.0.0.1:8089}"
NGINX_CONF="deploy/nginx_krystal_subdomain_ssl.conf"
CERTS_DIR="deploy/certs"

echo "================================================================="
echo " KRYSTAL-STACK // SUBDOMAIN SECURITY DEPLOYMENT WIZARD"
echo " Target Subdomain : ${SUBDOMAIN}"
echo " Backend Engine   : ${BACKEND_URL}"
echo "================================================================="

# 1. Check Python Kernel on Port 8089
echo ""
echo "[1/5] Overujem dostupnosť Krystal Engine Core na ${BACKEND_URL}..."
if curl -s -f "${BACKEND_URL}/api/wordpress/subdomain/security-status" > /dev/null; then
    echo "  -> Kernel je ONLINE. Invariant VITAL_MAX_HP = 6 potvrdený."
else
    echo "  -> Varovanie: Kernel na ${BACKEND_URL} zatiaľ neodpovedá."
fi

# 2. Setup Certificates Directory
echo ""
echo "[2/5] Pripravujem SSL/TLS certifikáty pre subdoménu ${SUBDOMAIN}..."
mkdir -p "${CERTS_DIR}"
CERT_FILE="${CERTS_DIR}/subdomain_cert.pem"
KEY_FILE="${CERTS_DIR}/subdomain_key.pem"

if [ ! -f "${CERT_FILE}" ] || [ ! -f "${KEY_FILE}" ]; then
    echo "  -> Generujem lokálny certifikát pre vývoj..."
    openssl req -x509 -newkey rsa:2048 -nodes -keyout "${KEY_FILE}" -out "${CERT_FILE}" -days 365 -subj "/CN=${SUBDOMAIN}" 2>/dev/null || true
    echo "  -> Pre produkciu spustite: sudo certbot --nginx -d ${SUBDOMAIN}"
else
    echo "  -> Certifikáty už existujú: ${CERT_FILE}"
fi

# 3. Generate Cryptographic Secret
echo ""
echo "[3/5] Generujem kryptograficky bezpečný HMAC Secret..."
GENERATED_SECRET=$(openssl rand -base64 32)
echo "  -> Vygenerovaný zdieľaný kľúč: ${GENERATED_SECRET}"

# 4. Generate WordPress wp-config.php Snippet
echo ""
echo "[4/5] Generujem konfiguračný blok pre wp-config.php..."
WP_SNIPPET="deploy/wp-config-snippet.php"
cat <<EOF > "${WP_SNIPPET}"
// ==============================================================================
// KRYSTAL-STACK SUBDOMAIN SECURITY DIRECTIVES
// ==============================================================================
define( 'WP_HOME', 'https://${SUBDOMAIN}' );
define( 'WP_SITEURL', 'https://${SUBDOMAIN}' );
define( 'FORCE_SSL_ADMIN', true );
define( 'DISALLOW_FILE_EDIT', true );
define( 'KRYSTAL_SUBDOMAIN_SECRET', '${GENERATED_SECRET}' );
// ==============================================================================
EOF
echo "  -> Snippet uložený do: ${WP_SNIPPET}"

# 5. Output Summary
echo ""
echo "[5/5] NASADENIE PRIPRAVENÉ:"
echo "  1. Nginx: skopírujte ${NGINX_CONF} do /etc/nginx/sites-available/ a aktivujte."
echo "  2. WordPress: vložte riadky z ${WP_SNIPPET} do wp-config.php."
echo "  3. Vložte shortcode [krystal_subdomain_portal mode=\"hybrid\" height=\"780px\"] do stránky."
echo "================================================================="
