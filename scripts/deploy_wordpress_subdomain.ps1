# ==============================================================================
# KRYSTAL-STACK: WORDPRESS SUBDOMAIN AUTOMATED DEPLOYMENT SCRIPT (PowerShell)
# ==============================================================================
param (
    [string]$Subdomain = "krystal.poslednikmen.cz",
    [string]$BackendUrl = "http://127.0.0.1:8089",
    [string]$NginxConfigPath = "deploy\nginx_krystal_subdomain_ssl.conf",
    [string]$CertsDir = "deploy\certs"
)

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host " KRYSTAL-STACK // SUBDOMAIN SECURITY DEPLOYMENT WIZARD" -ForegroundColor Yellow
Write-Host " Target Subdomain : $Subdomain" -ForegroundColor White
Write-Host " Backend Engine   : $BackendUrl" -ForegroundColor White
Write-Host "=================================================================" -ForegroundColor Cyan

# 1. Check Python Kernel on Port 8089
Write-Host "`n[1/5] Overujem dostupnost Krystal Engine Core na $BackendUrl..." -ForegroundColor Cyan
try {
    $response = Invoke-RestMethod -Uri "$BackendUrl/api/wordpress/subdomain/security-status" -Method Get -TimeoutSec 3
    if ($response.success -eq $true -and $response.vital_max_hp_rule -eq 6) {
        Write-Host "  -> Kernel je ONLINE. Invariant VITAL_MAX_HP = 6 potvrdeny." -ForegroundColor Green
    } else {
        Write-Host "  -> Varovanie: Odpoved z $BackendUrl nie je v sulade s ocakavanim." -ForegroundColor Yellow
    }
} catch {
    Write-Host "  -> Upozornenie: Kernel na $BackendUrl zatial neodpoveda. Uistite sa, ze bezi 'python -m krystal_web_hub.krystal_engine_core 8089'." -ForegroundColor Yellow
}

# 2. Check / Generate Certificates Directory
Write-Host "`n[2/5] Pripravujem SSL/TLS certifikaty pre subdomenu $Subdomain..." -ForegroundColor Cyan
if (-not (Test-Path $CertsDir)) {
    New-Item -ItemType Directory -Path $CertsDir -Force | Out-Null
    Write-Host "  -> Adresar $CertsDir vytvoreny." -ForegroundColor Gray
}

$certFile = Join-Path $CertsDir "subdomain_cert.pem"
$keyFile  = Join-Path $CertsDir "subdomain_key.pem"

if (-not (Test-Path $certFile) -or -not (Test-Path $keyFile)) {
    Write-Host "  -> Generujem lokalny self-signed certifikat pre testovanie..." -ForegroundColor Gray
    try {
        openssl req -x509 -newkey rsa:2048 -nodes -keyout $keyFile -out $certFile -days 365 -subj "/CN=$Subdomain" 2>$null
        Write-Host "  -> Certifikaty uspesne vytvorene v $CertsDir." -ForegroundColor Green
    } catch {
        Write-Host "  -> OpenSSL nie je priamo dostupny. Pre produkciu pouzite 'certbot --nginx -d $Subdomain'." -ForegroundColor Yellow
    }
} else {
    Write-Host "  -> Certifikaty uz existuju: $certFile" -ForegroundColor Green
}

# 3. Generate Cryptographic HMAC Secret
Write-Host "`n[3/5] Generujem kryptograficky bezpecny HMAC Secret..." -ForegroundColor Cyan
$rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
$bytes = New-Object byte[] 32
$rng.GetBytes($bytes)
$generatedSecret = [System.Convert]::ToBase64String($bytes)
Write-Host "  -> Vygenerovany zdielany kluc (HMAC SHA-256): $generatedSecret" -ForegroundColor Green

# 4. Generate WordPress wp-config.php Snippet
Write-Host "`n[4/5] Generujem konfiguracny blok pre wp-config.php..." -ForegroundColor Cyan
$wpSnippet = @"
// ==============================================================================
// KRYSTAL-STACK SUBDOMAIN SECURITY DIRECTIVES
// ==============================================================================
define( 'WP_HOME', 'https://$Subdomain' );
define( 'WP_SITEURL', 'https://$Subdomain' );
define( 'FORCE_SSL_ADMIN', true );
define( 'DISALLOW_FILE_EDIT', true );
define( 'KRYSTAL_SUBDOMAIN_SECRET', '$generatedSecret' );
// ==============================================================================
"@

$wpConfigFile = "deploy\wp-config-snippet.php"
Set-Content -Path $wpConfigFile -Value $wpSnippet
Write-Host "  -> Snippet ulozeny do: $wpConfigFile" -ForegroundColor Green

# 5. Output Summary & Instructions
Write-Host "`n[5/5] NASADENIE PRIPRAVENE // SMRZ ZHRNUTIE:" -ForegroundColor Yellow
Write-Host @"
1. NGINX:
   - Konfiguracia: $NginxConfigPath
   - Prelinkujte do /etc/nginx/sites-enabled/ a spustite: nginx -t && nginx -s reload

2. WORDPRESS:
   - Skopirujte obsah '$wpConfigFile' do vasho wp-config.php
   - Aktivujte plugin 'krystal-posledni-kmen' v admin paneli WordPress
   - Na stranke pouzite shortcode: [krystal_subdomain_portal mode="hybrid" height="780px"]

3. INTERAKTIVNE STUDIO:
   - Spustite v prehliadaci: http://localhost:8089/wordpress-subdomain-security
"@ -ForegroundColor White

Write-Host "=================================================================" -ForegroundColor Cyan
