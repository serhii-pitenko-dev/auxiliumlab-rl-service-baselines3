# Quick command aliases for common tasks

# Activate virtual environment
Write-Host "Quick Commands - Source this file to enable:" -ForegroundColor Cyan
Write-Host "  . .\scripts\aliases.ps1" -ForegroundColor Yellow
Write-Host ""

function grpc-gen {
    python -m grpc_tools.protoc -I./proto --python_out=./generated --grpc_python_out=./generated proto/policy_trainer.proto
}

function test {
    pytest -v
}

function test-cov {
    pytest --cov=auxilium_rl --cov-report=html --cov-report=term-missing -v
}

function serve {
    python server.py
}

function clean {
    Get-ChildItem -Path . -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    Get-ChildItem -Path . -Recurse -Directory -Filter .pytest_cache | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -Path .coverage, htmlcov, test_checkpoints, test_models -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "Cleaned!" -ForegroundColor Green
}

Write-Host "Available commands:" -ForegroundColor Green
Write-Host "  grpc-gen   - Generate gRPC code"
Write-Host "  test       - Run tests"
Write-Host "  test-cov   - Run tests with coverage"
Write-Host "  serve      - Start server"
Write-Host "  clean      - Clean artifacts"
Write-Host ""
