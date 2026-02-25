# PowerShell Build Scripts for RL Training Service

# Generate gRPC code from proto files
function Generate-GrpcCode {
    Write-Host "Generating gRPC code from proto files..." -ForegroundColor Cyan
    python -m grpc_tools.protoc -I./proto --python_out=./generated --grpc_python_out=./generated proto/policy_trainer.proto
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "gRPC code generated successfully!" -ForegroundColor Green
    } else {
        Write-Host "Failed to generate gRPC code" -ForegroundColor Red
        exit 1
    }
}

# Run tests with coverage
function Run-Tests {
    Write-Host "Running tests with coverage..." -ForegroundColor Cyan
    pytest --cov=auxilium_rl --cov-report=html --cov-report=term-missing -v
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Tests passed!" -ForegroundColor Green
        Write-Host "Coverage report available at: htmlcov/index.html" -ForegroundColor Yellow
    } else {
        Write-Host "Tests failed!" -ForegroundColor Red
        exit 1
    }
}

# Run tests without coverage (faster)
function Run-TestsFast {
    Write-Host "Running tests (fast mode)..." -ForegroundColor Cyan
    pytest -v
}

# Clean build artifacts and cache files
function Clean-BuildArtifacts {
    Write-Host "Cleaning build artifacts..." -ForegroundColor Cyan
    
    # Remove Python cache
    Get-ChildItem -Path . -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    Get-ChildItem -Path . -Recurse -Directory -Filter .pytest_cache | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    
    # Remove coverage reports
    Remove-Item -Path .coverage -Force -ErrorAction SilentlyContinue
    Remove-Item -Path htmlcov -Recurse -Force -ErrorAction SilentlyContinue
    
    # Remove test artifacts
    Remove-Item -Path test_checkpoints -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -Path test_models -Recurse -Force -ErrorAction SilentlyContinue
    
    Write-Host "Cleanup complete!" -ForegroundColor Green
}

# Setup development environment
function Setup-DevEnvironment {
    Write-Host "Setting up development environment..." -ForegroundColor Cyan
    
    # Check if virtual environment exists
    if (-not (Test-Path .venv)) {
        Write-Host "Creating virtual environment..." -ForegroundColor Yellow
        python -m venv .venv
    }
    
    # Activate virtual environment
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & .venv\Scripts\Activate.ps1
    
    # Install dependencies
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    pip install -r requirements.txt
    
    # Generate gRPC code
    Generate-GrpcCode
    
    Write-Host "Development environment ready!" -ForegroundColor Green
}

# Start the gRPC server
function Start-Server {
    Write-Host "Starting gRPC server..." -ForegroundColor Cyan
    python server.py
}

# Run a complete build (generate code + run tests)
function Build-All {
    Write-Host "Running complete build..." -ForegroundColor Cyan
    Generate-GrpcCode
    Run-Tests
}

# Show available commands
function Show-Help {
    Write-Host ""
    Write-Host "Available Commands:" -ForegroundColor Cyan
    Write-Host "  Generate-GrpcCode       - Generate gRPC code from proto files"
    Write-Host "  Run-Tests               - Run tests with coverage"
    Write-Host "  Run-TestsFast           - Run tests without coverage (faster)"
    Write-Host "  Clean-BuildArtifacts    - Clean build artifacts and cache"
    Write-Host "  Setup-DevEnvironment    - Setup development environment"
    Write-Host "  Start-Server            - Start the gRPC server"
    Write-Host "  Build-All               - Generate code and run tests"
    Write-Host "  Show-Help               - Show this help message"
    Write-Host ""
    Write-Host "Usage Examples:" -ForegroundColor Yellow
    Write-Host "  . .\scripts\build.ps1; Generate-GrpcCode"
    Write-Host "  . .\scripts\build.ps1; Run-Tests"
    Write-Host "  . .\scripts\build.ps1; Start-Server"
    Write-Host ""
}

# Export functions (if script is dot-sourced)
Export-ModuleMember -Function @(
    'Generate-GrpcCode',
    'Run-Tests',
    'Run-TestsFast',
    'Clean-BuildArtifacts',
    'Setup-DevEnvironment',
    'Start-Server',
    'Build-All',
    'Show-Help'
)

# Show help by default
Show-Help
