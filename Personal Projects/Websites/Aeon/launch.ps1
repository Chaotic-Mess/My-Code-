# AEON SERVICES - LAUNCH SCRIPT
# Choose your weapon:

Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "ÆEON SERVICES - DEPLOYMENT PROTOCOL" -ForegroundColor Green
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

$choice = Read-Host "Select server type:
[1] Python (http.server)
[2] Node.js (npx http-server)
[3] PHP (built-in server)

Enter choice"

switch ($choice) {
    "1" {
        Write-Host "`nInitializing Python server on port 8000..." -ForegroundColor Yellow
        Write-Host "Access at: http://localhost:8000" -ForegroundColor Green
        python -m http.server 8000
    }
    "2" {
        Write-Host "`nInitializing Node.js server on port 8080..." -ForegroundColor Yellow
        Write-Host "Access at: http://localhost:8080" -ForegroundColor Green
        npx http-server -p 8080
    }
    "3" {
        Write-Host "`nInitializing PHP server on port 8000..." -ForegroundColor Yellow
        Write-Host "Access at: http://localhost:8000" -ForegroundColor Green
        php -S localhost:8000
    }
    default {
        Write-Host "`nInvalid choice. Defaulting to Python..." -ForegroundColor Red
        python -m http.server 8000
    }
}
