# Generate gRPC code for both proto files
Write-Host "Generating Python gRPC code..." -ForegroundColor Cyan

# Generate policy_trainer.proto
Write-Host "`nGenerating policy_trainer.proto..." -ForegroundColor Yellow
python -m grpc_tools.protoc -I./proto --python_out=./generated --grpc_python_out=./generated proto/policy_trainer.proto

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ policy_trainer.proto generated successfully" -ForegroundColor Green
} else {
    Write-Host "✗ Failed to generate policy_trainer.proto" -ForegroundColor Red
    exit 1
}

# Generate simulation.proto
Write-Host "`nGenerating simulation.proto..." -ForegroundColor Yellow
python -m grpc_tools.protoc -I./proto --python_out=./generated --grpc_python_out=./generated proto/simulation.proto

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ simulation.proto generated successfully" -ForegroundColor Green
} else {
    Write-Host "✗ Failed to generate simulation.proto" -ForegroundColor Red
    exit 1
}

Write-Host "`n✓ All gRPC code generated successfully!" -ForegroundColor Green
Write-Host "`nGenerated files in ./generated/:" -ForegroundColor Cyan
Get-ChildItem ./generated/*.py | ForEach-Object { Write-Host "  - $($_.Name)" }
