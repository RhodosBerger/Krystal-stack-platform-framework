
Write-Host "=== FANUC RISE: UNIVERSAL REPAIR TOOL (Vue + React) ==="
    
# Function to repair a directory
function Repair-Frontend($path) {
    Write-Host ">>> REPAIRING: $path"
    if (Test-Path $path) {
        Push-Location $path
        
        Write-Host "  - Stopping Node..."
        Stop-Process -Name "node" -ErrorAction SilentlyContinue
        
        Write-Host "  - Cleaning Modules..."
        if (Test-Path "node_modules") { Remove-Item -Recurse -Force "node_modules" -ErrorAction SilentlyContinue }
        if (Test-Path "package-lock.json") { Remove-Item -Force "package-lock.json" -ErrorAction SilentlyContinue }
        
        Write-Host "  - Cleaning Cache..."
        npm cache clean --force
        
        Write-Host "  - Installing..."
        npm install
        
        # Apply specific Windows fix for Vue (Rollup)
        if ($path -like "*frontend-vue*") {
            Write-Host "  - Applying Vue/Rollup Windows Patch..."
            npm install --save-optional @rollup/rollup-win32-x64-msvc
        }
        
        Pop-Location
        Write-Host ">>> DONE: $path"
    }
    else {
        Write-Warning "Directory not found: $path"
    }
}

# Run Repair on both
Repair-Frontend ".\frontend-vue"
Repair-Frontend ".\frontend-react"

Write-Host "=== ALL REPAIRS COMPLETE ==="
Write-Host "You can now run 'npm run dev' in each folder, or use start_dev.ps1."

