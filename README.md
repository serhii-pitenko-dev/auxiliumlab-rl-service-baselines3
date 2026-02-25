# AISandbox RL Training Service

A **gRPC service** for training Reinforcement Learning agents using **Stable Baselines3**. Supports PPO, A2C, and DQN algorithms. Integrates bidirectionally with the .NET `AuxiliumLab.AiSandbox` simulation engine.

## Role in the System

```
.NET AuxiliumLab.AiSandbox
  AiTrainingOrchestrator
    PolicyTrainerClient  ──── gRPC :50051 ────►  This Python service
                                                    TrainingOrchestrator
                                                      ExternalSimEnv
                                                        GrpcExternalEnvAdapter
                                                          └─── gRPC :50062 ────►  .NET GrpcHost
                                                                                      SimulationService
                                                                                       (gym reset/step)
```

- **.NET → Python (port 50051):** .NET calls `StartTrainingPPO/A2C/DQN`, `GetTrainingStatus`, `Act`.
- **Python → .NET (port 50062):** Python gym calls `reset` and `step` on the C# simulation during training.

## Architecture

```
auxilium_rl/
├── transport/          gRPC server layer
│   ├── grpc_server.py      Server factory and startup
│   ├── trainer_servicer.py gRPC handler for PolicyTrainerService RPCs
│   └── health_servicer.py  gRPC health check protocol
├── core/               Business logic (no transport concerns)
│   ├── training.py         TrainingOrchestrator + CheckpointCallback
│   ├── algorithms.py       SB3 model factory (PPO / A2C / DQN)
│   ├── env.py              ExternalSimEnv (gymnasium.Env wrapper)
│   └── dto.py              TrainingConfig, RunInfo, RunStatus, AlgorithmType
└── infra/              Infrastructure
    ├── config.py           ServiceConfig and EnvConfig (from environment variables)
    ├── external_env_adapter.py  ExternalEnvAdapter ABC + FakeAdapter + GrpcAdapter
    ├── model_store.py      Model/checkpoint save & load (zip format)
    └── logging.py          Logging setup
```

## Key Components

### `TrainingOrchestrator` (`core/training.py`)
Thread-safe manager for multiple concurrent training runs.
- `start_training(config, adapter_factory)` — starts training in a background thread, returns `run_id`.
- `get_run_status(run_id)` — returns `RunInfo` (timesteps done, status, last checkpoint path).
- `get_model(run_id)` — returns the trained `BaseAlgorithm` for inference.

Training uses `CheckpointCallback` to save intermediate checkpoints every `checkpoint_freq` steps (default 10 000).

### `ExternalSimEnv` (`core/env.py`)
Standard `gymnasium.Env` that delegates all `reset()` / `step()` calls to an `ExternalEnvAdapter`.

| Space | Type | Default shape |
|---|---|---|
| `observation_space` | `Box(−∞, +∞)` | `(4,)` — position + stats |
| `action_space` | `Discrete` | `4` — Move N/S/E/W (or 0…3) |

`max_steps` controls episode truncation (default 500).

### `ExternalEnvAdapter` (`infra/external_env_adapter.py`)
Adapter interface between the gym and the simulation backend.

| Implementation | Used when |
|---|---|
| `FakeExternalEnvAdapter` | Unit tests and local development without a .NET process |
| `GrpcExternalEnvAdapter` | Production training against the live .NET `GrpcHost` (:50062) |

The adapter factory is passed to `create_server` and a new adapter instance is created per training run.

### `ModelStore` (`infra/model_store.py`)
Handles model persistence:
- `save_model(model, run_id)` — saves to `{models_dir}/{run_id}/final.zip`.
- `save_checkpoint(model, run_id, step)` — saves to `{checkpoint_dir}/{run_id}/step_{step}.zip`.
- `load_model(run_id, algorithm)` — loads and returns the model; requires the algorithm type for correct class instantiation.

### `trainer_servicer.py` (Transport)
Implements `PolicyTrainerServiceServicer`:

| RPC | Handler |
|---|---|
| `StartTrainingPPO` | Starts PPO run via `TrainingOrchestrator` |
| `StartTrainingA2C` | Starts A2C run |
| `StartTrainingDQN` | Starts DQN run |
| `GetTrainingStatus` | Returns progress from run registry |
| `Act` | Loads model and runs `model.predict(observation)` |

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

### 3. Generate gRPC Code (if proto changed)
```powershell
python -m grpc_tools.protoc `
  -I./proto `
  --python_out=./generated `
  --grpc_python_out=./generated `
  proto/policy_trainer.proto
```
Or use the helper script:
```powershell
.\scripts\generate_all_grpc.ps1
```

## Running

```powershell
python server.py        # starts gRPC server on :50051
```

### Environment Variables
| Variable | Default | Description |
|---|---|---|
| `GRPC_HOST` | `0.0.0.0` | Bind address |
| `GRPC_PORT` | `50051` | Listen port |
| `MODELS_DIR` | `./trained_models` | Final model storage |
| `CHECKPOINT_DIR` | `./checkpoints` | Checkpoint storage |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `OBSERVATION_DIM` | `4` | Observation vector size |
| `ACTION_DIM` | `4` | Number of discrete actions |
| `MAX_STEPS` | `500` | Max steps per episode |

## Testing

```powershell
pytest                                         # all tests
pytest tests/test_algorithms.py -v            # algorithm factory
pytest tests/test_env_wrapper.py -v           # gym environment
pytest tests/test_grpc_training_smoke.py -v   # end-to-end smoke
pytest tests/test_health_check.py -v          # health check protocol
```

## API Usage

### Start Training (PPO example)
```python
import grpc
from generated import policy_trainer_pb2, policy_trainer_pb2_grpc

channel = grpc.insecure_channel('localhost:50051')
stub = policy_trainer_pb2_grpc.PolicyTrainerServiceStub(channel)

response = stub.StartTrainingPPO(policy_trainer_pb2.TrainingRequest(
    experiment_id="run_001",
    total_timesteps=100_000,
    seed=42,
    hyperparameters={"learning_rate": "3e-4", "n_steps": "2048"},
    model_output_path="./trained_models/run_001.zip"
))
run_id = response.run_id
```

### Poll Status
```python
status = stub.GetTrainingStatus(policy_trainer_pb2.StatusRequest(run_id=run_id))
print(f"Steps done: {status.timesteps_done} | Done: {status.is_done}")
```

### Inference
```python
act = stub.Act(policy_trainer_pb2.ActRequest(
    run_id=run_id,
    observation=[0.1, 0.2, 0.3, 0.4]
))
print(f"Action: {act.action}")
```

## Default Hyperparameters

| Algorithm | Key defaults |
|---|---|
| PPO | `learning_rate=3e-4`, `n_steps=2048`, `batch_size=64`, `n_epochs=10`, `gamma=0.99` |
| A2C | `learning_rate=7e-4`, `n_steps=5`, `gamma=0.99`, `vf_coef=0.5` |
| DQN | `learning_rate=1e-4`, `buffer_size=50000`, `batch_size=32`, `gamma=0.99` |

Override any value via the `hyperparameters` map in `TrainingRequest`.

## Proto Files

```
proto/
├── policy_trainer.proto  .NET → Python: start training, get status, act
└── simulation.proto      Python → .NET: gym reset / step / close  (shared with GrpcHost)
```

The generated stubs in `generated/` are auto-generated and should not be edited manually.

## Adding a New Algorithm
1. Add a new value to the `AlgorithmType` enum in `core/dto.py`.
2. Add the SB3 import and a branch in `build_model()` in `core/algorithms.py`.
3. Add a corresponding `StartTrainingXxx` RPC to `proto/policy_trainer.proto`.
4. Regenerate stubs and add a handler in `transport/trainer_servicer.py`.
5. Add the RPC implementation to `IPolicyTrainerClient` in the .NET `AiTrainingOrchestrator`.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'generated'` | Run protoc command; ensure `generated/__init__.py` exists |
| `gRPC Connection refused :50051` | Start `python server.py` first |
| `gRPC Connection refused :50062` | Start the .NET GrpcHost (Training mode) first |
| Training too slow | Reduce `total_timesteps`; check `max_steps` per episode |
| `LOG_LEVEL=DEBUG` | Set env var for verbose output |


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
