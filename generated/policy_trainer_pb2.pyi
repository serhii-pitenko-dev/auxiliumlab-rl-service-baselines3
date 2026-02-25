"""Type stubs for generated protobuf code."""
from typing import Any, ClassVar, Iterable, Mapping
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper

DESCRIPTOR: _descriptor.FileDescriptor

class TrainingStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    STARTED: TrainingStatus
    FAILED: TrainingStatus

STARTED: TrainingStatus
FAILED: TrainingStatus

class TrainingRequest(_message.Message):
    
    class HyperparametersEntry(_message.Message):
        KEY_FIELD_NUMBER: ClassVar[int]
        VALUE_FIELD_NUMBER: ClassVar[int]
        key: str
        value: str
        def __init__(self, key: str = ..., value: str = ...) -> None: ...
    
    EXPERIMENT_ID_FIELD_NUMBER: ClassVar[int]
    TOTAL_TIMESTEPS_FIELD_NUMBER: ClassVar[int]
    SEED_FIELD_NUMBER: ClassVar[int]
    HYPERPARAMETERS_FIELD_NUMBER: ClassVar[int]
    MODEL_OUTPUT_PATH_FIELD_NUMBER: ClassVar[int]
    experiment_id: str
    total_timesteps: int
    seed: int
    hyperparameters: _containers.ScalarMap[str, str]
    model_output_path: str
    def __init__(
        self,
        experiment_id: str = ...,
        total_timesteps: int = ...,
        seed: int = ...,
        hyperparameters: Mapping[str, str] | None = ...,
        model_output_path: str = ...,
    ) -> None: ...

class TrainingResponse(_message.Message):
    STATUS_FIELD_NUMBER: ClassVar[int]
    MESSAGE_FIELD_NUMBER: ClassVar[int]
    RUN_ID_FIELD_NUMBER: ClassVar[int]
    status: TrainingStatus
    message: str
    run_id: str
    def __init__(
        self,
        status: TrainingStatus = ...,
        message: str = ...,
        run_id: str = ...,
    ) -> None: ...

class StatusRequest(_message.Message):
    RUN_ID_FIELD_NUMBER: ClassVar[int]
    run_id: str
    def __init__(self, run_id: str = ...) -> None: ...

class StatusResponse(_message.Message):
    TIMESTEPS_DONE_FIELD_NUMBER: ClassVar[int]
    IS_DONE_FIELD_NUMBER: ClassVar[int]
    LAST_CHECKPOINT_PATH_FIELD_NUMBER: ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: ClassVar[int]
    timesteps_done: int
    is_done: bool
    last_checkpoint_path: str
    error_message: str
    def __init__(
        self,
        timesteps_done: int = ...,
        is_done: bool = ...,
        last_checkpoint_path: str = ...,
        error_message: str = ...,
    ) -> None: ...

class ActRequest(_message.Message):
    RUN_ID_FIELD_NUMBER: ClassVar[int]
    OBSERVATION_FIELD_NUMBER: ClassVar[int]
    run_id: str
    observation: _containers.RepeatedScalarFieldContainer[float]
    def __init__(
        self,
        run_id: str = ...,
        observation: Iterable[float] | None = ...,
    ) -> None: ...

class ActResponse(_message.Message):
    ACTION_FIELD_NUMBER: ClassVar[int]
    SUCCESS_FIELD_NUMBER: ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: ClassVar[int]
    action: int
    success: bool
    error_message: str
    def __init__(
        self,
        action: int = ...,
        success: bool = ...,
        error_message: str = ...,
    ) -> None: ...
