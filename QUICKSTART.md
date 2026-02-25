# Quick Start Guide

## 5-Minute Setup

### 1. Activate Virtual Environment
```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Generate gRPC Code (Already Done!)
The gRPC code has been generated. To regenerate if needed:
```powershell
python -m grpc_tools.protoc -I./proto --python_out=./generated --grpc_python_out=./generated proto/policy_trainer.proto
```

### 3. Start the Server
```powershell
python server.py
```

You should see:
```
INFO - Starting RL Training Service...
INFO - gRPC server started on 0.0.0.0:50051
INFO - Server is ready to accept requests
```

### 4. Run the Example Client (in another terminal)
```powershell
# Activate venv in new terminal
.\.venv\Scripts\Activate.ps1

# Run example
python example_client.py
```

### 5. Check Server Health (Optional)
```powershell
# Check if server is healthy
python healthcheck.py

# Or specify custom host/port
python healthcheck.py --host localhost --port 50051
```

The health check verifies:
- ✓ Server is running and accepting connections
- ✓ Training orchestrator is available
- ✓ Model storage directories are accessible

## Health Monitoring

The gRPC server implements the standard [gRPC Health Checking Protocol](https://github.com/grpc/grpc/blob/master/doc/health-checking.md).

### Using the Health Check Client
```powershell
# Basic health check
python healthcheck.py

# Verbose output
python healthcheck.py --verbose

# Custom timeout
python healthcheck.py --timeout 10
```

### Using grpcurl for Health Checks
```bash
# Install grpcurl (one time)
# See: https://github.com/fullstorydev/grpcurl

# Check health
grpcurl -plaintext localhost:50051 grpc.health.v1.Health/Check

# Should return:
# {
#   "status": "SERVING"
# }
```

### Integration with Monitoring Tools

The health check can be used with:
- **Kubernetes**: Configure `livenessProbe` and `readinessProbe`
- **Docker Compose**: Use `healthcheck` directive
- **Load Balancers**: Configure health check endpoints
- **Monitoring Systems**: Prometheus, Datadog, etc.

Example Kubernetes probe configuration:
```yaml
livenessProbe:
  exec:
    command: ["/bin/grpc_health_probe", "-addr=:50051"]
  initialDelaySeconds: 5
readinessProbe:
  exec:
    command: ["/bin/grpc_health_probe", "-addr=:50051"]
  initialDelaySeconds: 10
```

## Testing

### Run All Tests
```powershell
pytest -v
```

### Run Tests with Coverage
```powershell
pytest --cov=auxilium_rl --cov-report=html -v
```

## Using the PowerShell Build Scripts

### Option 1: Makefile-style
```powershell
# Show available commands
pwsh -File Makefile.ps1

# Run tests
pwsh -File Makefile.ps1 test

# Start server
pwsh -File Makefile.ps1 serve
```

### Option 2: Source build functions
```powershell
# Load functions
. .\scripts\build.ps1

# Use them
Generate-GrpcCode
Run-Tests
Start-Server
```

### Option 3: Quick aliases
```powershell
# Load aliases
. .\scripts\aliases.ps1

# Use short commands
grpc-gen    # Generate code
test        # Run tests
test-cov    # Run with coverage
serve       # Start server
clean       # Clean artifacts
```

## Troubleshooting

### "No module named 'generated'"
- Make sure you generated the gRPC code
- Run from project root directory

### "Module 'grpc_tools' not found"
- Activate the virtual environment: `.\.venv\Scripts\Activate.ps1`
- Check installation: `pip show grpcio-tools`

### Server won't start
- Check if port 50051 is available
- Check firewall settings
- Try a different port: `$env:GRPC_PORT = "50052"; python server.py`

## Project Structure Cheat Sheet

```
proto/                    # Protocol buffer definitions
generated/                # Auto-generated gRPC code
auxilium_rl/
  ├── transport/          # gRPC server & servicer
  ├── core/              # Business logic (algorithms, training, env)
  └── infra/             # Infrastructure (config, storage, adapters)
server.py                 # Main entry point
example_client.py         # Usage example
tests/                    # Unit and integration tests
```

## Next Steps

1. ✓ Project created
2. ✓ gRPC code generated
3. ⏭ Start the server: `python server.py`
4. ⏭ Run tests: `pytest -v`
5. ⏭ Try the example: `python example_client.py`
6. ⏭ Customize environment adapter for your .NET simulation
7. ⏭ Adjust hyperparameters as needed
8. ⏭ Add your own experiments!
