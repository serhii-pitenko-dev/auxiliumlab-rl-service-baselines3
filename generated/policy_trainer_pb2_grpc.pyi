"""Type stubs for generated gRPC service code."""
import grpc
from . import policy_trainer_pb2

class PolicyTrainerServiceStub:
    """Service for training and inference with RL agents"""
    def __init__(self, channel: grpc.Channel) -> None: ...
    
    StartTrainingPPO: grpc.UnaryUnaryMultiCallable[
        policy_trainer_pb2.TrainingRequest,
        policy_trainer_pb2.TrainingResponse
    ]
    
    StartTrainingA2C: grpc.UnaryUnaryMultiCallable[
        policy_trainer_pb2.TrainingRequest,
        policy_trainer_pb2.TrainingResponse
    ]
    
    StartTrainingDQN: grpc.UnaryUnaryMultiCallable[
        policy_trainer_pb2.TrainingRequest,
        policy_trainer_pb2.TrainingResponse
    ]
    
    GetTrainingStatus: grpc.UnaryUnaryMultiCallable[
        policy_trainer_pb2.StatusRequest,
        policy_trainer_pb2.StatusResponse
    ]
    
    Act: grpc.UnaryUnaryMultiCallable[
        policy_trainer_pb2.ActRequest,
        policy_trainer_pb2.ActResponse
    ]

class PolicyTrainerServiceServicer:
    """Service for training and inference with RL agents"""
    def StartTrainingPPO(
        self,
        request: policy_trainer_pb2.TrainingRequest,
        context: grpc.ServicerContext
    ) -> policy_trainer_pb2.TrainingResponse: ...
    
    def StartTrainingA2C(
        self,
        request: policy_trainer_pb2.TrainingRequest,
        context: grpc.ServicerContext
    ) -> policy_trainer_pb2.TrainingResponse: ...
    
    def StartTrainingDQN(
        self,
        request: policy_trainer_pb2.TrainingRequest,
        context: grpc.ServicerContext
    ) -> policy_trainer_pb2.TrainingResponse: ...
    
    def GetTrainingStatus(
        self,
        request: policy_trainer_pb2.StatusRequest,
        context: grpc.ServicerContext
    ) -> policy_trainer_pb2.StatusResponse: ...
    
    def Act(
        self,
        request: policy_trainer_pb2.ActRequest,
        context: grpc.ServicerContext
    ) -> policy_trainer_pb2.ActResponse: ...

def add_PolicyTrainerServiceServicer_to_server(
    servicer: PolicyTrainerServiceServicer,
    server: grpc.Server
) -> None: ...
