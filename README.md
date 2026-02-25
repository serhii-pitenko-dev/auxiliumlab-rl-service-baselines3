# AISandbox RL Training Service

A production-ready gRPC service for training Reinforcement Learning agents using Stable Baselines3. Supports PPO, A2C, and DQN algorithms with external environment integration.

## Features

- **gRPC API** for training and inference
- **Three RL Algorithms**: PPO, A2C, DQN
- **External Environment Integration**: Communicates with .NET simulation via adapter interface
- **Asynchronous Training**: Non-blocking training with progress tracking
- **Model Persistence**: Automatic checkpointing and model saving
- **Health Checks**: Built-in gRPC health checking protocol support
- **Production Architecture**: Clean separation of transport/core/infrastructure layers
- **Comprehensive Tests**: Unit and integration tests with pytest

## Project Structure

```
auxilium_rl_service_baselines3/
├── proto/
│   └── policy_trainer.proto          # gRPC service definition
├── generated/                         # Generated gRPC code (auto-generated)
├── auxilium_rl/
│   ├── transport/                    # gRPC layer
│   │   ├── grpc_server.py
│   │   ├── trainer_servicer.py
│   │   └── health_servicer.py        # Health check implementation
│   ├── core/                         # Business logic
│   │   ├── algorithms.py             # Model factory (PPO/A2C/DQN)
│   │   ├── training.py               # Training orchestration
│   │   ├── env.py                    # Gymnasium env wrapper
│   │   └── dto.py                    # Data transfer objects
│   └── infra/                        # Infrastructure
│       ├── config.py                 # Configuration
│       ├── model_store.py            # Model persistence
│       ├── external_env_adapter.py   # Environment adapter interface
│       └── logging.py                # Logging setup
├── tests/
│   ├── test_algorithms.py
│   ├── test_env_wrapper.py
│   ├── test_grpc_training_smoke.py
│   └── test_health_check.py          # Health check tests
├── server.py                         # Entry point
├── healthcheck.py                    # Health check client
├── requirements.txt
├── pytest.ini
└── README.md
```

## Prerequisites

- Python 3.8+
- Windows (or Linux/macOS with minor script adjustments)

## Setup

### 1. Create Virtual Environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 3. Generate gRPC Code

```powershell
python -m grpc_tools.protoc -I./proto --python_out=./generated --grpc_python_out=./generated proto/policy_trainer.proto
```

## Running the Service

### Start the Server

```powershell
python server.py
```

The server will start on `localhost:50051` by default.

### Configuration

Set environment variables to customize:

```powershell
$env:GRPC_HOST = "0.0.0.0"
$env:GRPC_PORT = "50051"
$env:MODELS_DIR = "./trained_models"
$env:CHECKPOINT_DIR = "./checkpoints"
$env:LOG_LEVEL = "INFO"
```

## API Usage

### Starting Training

#### PPO
```python
import grpc
from generated import policy_trainer_pb2, policy_trainer_pb2_grpc

channel = grpc.insecure_channel('localhost:50051')
stub = policy_trainer_pb2_grpc.PolicyTrainerServiceStub(channel)

request = policy_trainer_pb2.TrainingRequest(
    experiment_id="my_experiment",
    total_timesteps=100000,
    seed=42,
    hyperparameters={
        "learning_rate": "3e-4",
        "n_steps": "2048",
        "batch_size": "64"
    },
    model_output_path="./my_model.zip"
)

response = stub.StartTrainingPPO(request)
print(f"Run ID: {response.run_id}")
```

#### A2C
```python
response = stub.StartTrainingA2C(request)
```

#### DQN
```python
response = stub.StartTrainingDQN(request)
```

### Checking Training Status

```python
status_request = policy_trainer_pb2.StatusRequest(run_id=response.run_id)
status = stub.GetTrainingStatus(status_request)

print(f"Timesteps done: {status.timesteps_done}")
print(f"Is done: {status.is_done}")
print(f"Last checkpoint: {status.last_checkpoint_path}")
```

### Performing Inference

```python
act_request = policy_trainer_pb2.ActRequest(
    run_id=run_id,
    observation=[0.1, 0.2, 0.3, 0.4]
)

act_response = stub.Act(act_request)
print(f"Action: {act_response.action}")
```

## Testing

### Run All Tests

```powershell
pytest
```

### Run with Coverage

```powershell
pytest --cov=auxilium_rl --cov-report=html --cov-report=term-missing
```

### Run Specific Test File

```powershell
pytest tests/test_algorithms.py -v
```

## External Environment Adapter

The service uses an adapter pattern to communicate with external simulations:

### Interface
```python
class ExternalEnvAdapter(ABC):
    def reset(self, seed: int = None) -> np.ndarray: ...
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]: ...
    def close(self) -> None: ...
```

### Implementations

1. **FakeExternalEnvAdapter** (default): In-memory fake for testing
2. **GrpcExternalEnvAdapter** (placeholder): gRPC client for .NET simulation

To use a custom adapter:

```python
from auxilium_rl.transport.grpc_server import create_server

def my_adapter_factory():
    return MyCustomAdapter(endpoint="localhost:5000")

server = create_server(
    orchestrator=orchestrator,
    config=config,
    adapter_factory=my_adapter_factory
)
```

## Hyperparameters

### PPO Defaults
- `learning_rate`: 3e-4
- `n_steps`: 2048
- `batch_size`: 64
- `n_epochs`: 10
- `gamma`: 0.99
- `clip_range`: 0.2

### A2C Defaults
- `learning_rate`: 7e-4
- `n_steps`: 5
- `gamma`: 0.99
- `vf_coef`: 0.5

### DQN Defaults
- `learning_rate`: 1e-4
- `buffer_size`: 50000
- `batch_size`: 32
- `gamma`: 0.99

Override via the `hyperparameters` map in the training request.

## Build Scripts (PowerShell)

### Generate gRPC Code

```powershell
# scripts/generate_grpc.ps1
python -m grpc_tools.protoc -I./proto --python_out=./generated --grpc_python_out=./generated proto/policy_trainer.proto
```

### Run Tests

```powershell
# scripts/run_tests.ps1
pytest --cov=auxilium_rl --cov-report=html
```

### Clean Build Artifacts

```powershell
# scripts/clean.ps1
Remove-Item -Recurse -Force generated/*.py -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force __pycache__, .pytest_cache, .coverage, htmlcov -ErrorAction SilentlyContinue
```

## Development Workflow

1. **Make changes** to proto or Python code
2. **Regenerate gRPC code** if proto changed:
   ```powershell
   python -m grpc_tools.protoc -I./proto --python_out=./generated --grpc_python_out=./generated proto/policy_trainer.proto
   ```
3. **Run tests**:
   ```powershell
   pytest -v
   ```
4. **Start server**:
   ```powershell
   python server.py
   ```

## Troubleshooting

### Import Errors for Generated Code

If you see `ModuleNotFoundError: No module named 'generated'`, ensure:
1. You've run the protoc command to generate code
2. The `generated/` directory contains `__init__.py`
3. You're running from the project root directory

### Training Doesn't Complete

- Check logs for errors: `$env:LOG_LEVEL = "DEBUG"`
- Reduce `total_timesteps` for faster testing
- Verify environment adapter is working correctly

### gRPC Connection Refused

- Ensure server is running: `python server.py`
- Check port is not blocked by firewall
- Verify correct host/port in client code

## License

MIT License - See LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Contact

For questions or issues, please open a GitHub issue.
