# Simple Makefile-style script for common tasks
# Usage: pwsh -File Makefile.ps1 <target>

param(
    [Parameter(Position=0)]
    [string]$Target = "help"
)

switch ($Target) {
    "generate" {
        Write-Host "Generating gRPC code..." -ForegroundColor Cyan
        python -m grpc_tools.protoc -I./proto --python_out=./generated --grpc_python_out=./generated proto/policy_trainer.proto
    }
    
    "test" {
        Write-Host "Running tests..." -ForegroundColor Cyan
        pytest -v
    }
    
    "test-coverage" {
        Write-Host "Running tests with coverage..." -ForegroundColor Cyan
        pytest --cov=auxilium_rl --cov-report=html --cov-report=term-missing -v
    }
    
    "serve" {
        Write-Host "Starting server..." -ForegroundColor Cyan
        # OBS_DIM = 5 scalar features + (2*SightRange+1)^2 vision cells = 5 + 121 = 126 (SightRange=5)
        # ACTION_DIM = 5: up, down, left, right, toggle-run
        $env:OBS_DIM = "126"
        $env:ACTION_DIM = "5"
        python server.py
    }
    
    "clean" {
        Write-Host "Cleaning artifacts..." -ForegroundColor Cyan
        Get-ChildItem -Path . -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        Get-ChildItem -Path . -Recurse -Directory -Filter .pytest_cache | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        Remove-Item -Path .coverage, htmlcov -Recurse -Force -ErrorAction SilentlyContinue
        Remove-Item -Path test_checkpoints, test_models -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host "Clean complete!" -ForegroundColor Green
    }
    
    "setup" {
        Write-Host "Setting up environment..." -ForegroundColor Cyan
        if (-not (Test-Path .venv)) {
            python -m venv .venv
        }
        & .venv\Scripts\Activate.ps1
        pip install -r requirements.txt
        & $PSCommandPath generate
        Write-Host "Setup complete!" -ForegroundColor Green
    }
    
    "all" {
        & $PSCommandPath generate
        & $PSCommandPath test
    }
    
    default {
        Write-Host ""
        Write-Host "Available targets:" -ForegroundColor Cyan
        Write-Host "  setup          - Setup development environment"
        Write-Host "  generate       - Generate gRPC code from proto"
        Write-Host "  test           - Run tests"
        Write-Host "  test-coverage  - Run tests with coverage"
        Write-Host "  serve          - Start the gRPC server"
        Write-Host "  clean          - Clean build artifacts"
        Write-Host "  all            - Generate and test"
        Write-Host ""
        Write-Host "Usage: pwsh -File Makefile.ps1 <target>" -ForegroundColor Yellow
        Write-Host ""
    }
}
