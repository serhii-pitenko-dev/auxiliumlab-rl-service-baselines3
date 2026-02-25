"""
Example client script demonstrating how to use the RL Training Service.

This script shows how to:
1. Start training a PPO agent
2. Poll for training status
3. Perform inference once training is complete
"""
import grpc
import time
from generated import policy_trainer_pb2, policy_trainer_pb2_grpc


def main():
    """Main client demonstration."""
    # Connect to server
    print("Connecting to gRPC server at localhost:50051...")
    channel = grpc.insecure_channel('localhost:50051')
    stub = policy_trainer_pb2_grpc.PolicyTrainerServiceStub(channel)
    
    try:
        # 1. Start PPO training
        print("\n=== Starting PPO Training ===")
        training_request = policy_trainer_pb2.TrainingRequest(
            experiment_id="example_ppo_experiment",
            total_timesteps=5000,  # Small number for quick demo
            seed=42,
            hyperparameters={
                "learning_rate": "3e-4",
                "n_steps": "256",
                "batch_size": "64",
                "n_epochs": "5"
            },
            model_output_path="./trained_models/example_ppo.zip"
        )
        
        response = stub.StartTrainingPPO(training_request)
        
        if response.status == policy_trainer_pb2.STARTED:
            print(f"✓ Training started successfully!")
            print(f"  Run ID: {response.run_id}")
            print(f"  Message: {response.message}")
            run_id = response.run_id
        else:
            print(f"✗ Failed to start training: {response.message}")
            return
        
        # 2. Poll for training status
        print("\n=== Monitoring Training Progress ===")
        while True:
            status_request = policy_trainer_pb2.StatusRequest(run_id=run_id)
            status = stub.GetTrainingStatus(status_request)
            
            print(f"Progress: {status.timesteps_done}/5000 timesteps", end='\r')
            
            if status.is_done:
                print("\n✓ Training completed!")
                if status.error_message:
                    print(f"  Error: {status.error_message}")
                else:
                    print(f"  Final timesteps: {status.timesteps_done}")
                    if status.last_checkpoint_path:
                        print(f"  Last checkpoint: {status.last_checkpoint_path}")
                break
            
            time.sleep(2)  # Poll every 2 seconds
        
        # 3. Perform inference
        if not status.error_message:
            print("\n=== Testing Inference ===")
            # Create a sample observation (4-dimensional for our default env)
            sample_observation = [0.1, -0.2, 0.3, -0.1]
            
            act_request = policy_trainer_pb2.ActRequest(
                run_id=run_id,
                observation=sample_observation
            )
            
            act_response = stub.Act(act_request)
            
            if act_response.success:
                print(f"✓ Inference successful!")
                print(f"  Observation: {sample_observation}")
                print(f"  Predicted action: {act_response.action}")
            else:
                print(f"✗ Inference failed: {act_response.error_message}")
        
        # 4. Demonstrate A2C training (optional)
        print("\n=== Starting A2C Training (Quick Demo) ===")
        a2c_request = policy_trainer_pb2.TrainingRequest(
            experiment_id="example_a2c_experiment",
            total_timesteps=2000,
            seed=123,
            hyperparameters={
                "learning_rate": "7e-4",
                "n_steps": "5"
            },
            model_output_path="./trained_models/example_a2c.zip"
        )
        
        a2c_response = stub.StartTrainingA2C(a2c_request)
        if a2c_response.status == policy_trainer_pb2.STARTED:
            print(f"✓ A2C training started with run ID: {a2c_response.run_id}")
            print("  (Training in background...)")
        
        # 5. Demonstrate DQN training (optional)
        print("\n=== Starting DQN Training (Quick Demo) ===")
        dqn_request = policy_trainer_pb2.TrainingRequest(
            experiment_id="example_dqn_experiment",
            total_timesteps=2000,
            seed=456,
            hyperparameters={
                "learning_rate": "1e-4",
                "buffer_size": "10000"
            },
            model_output_path="./trained_models/example_dqn.zip"
        )
        
        dqn_response = stub.StartTrainingDQN(dqn_request)
        if dqn_response.status == policy_trainer_pb2.STARTED:
            print(f"✓ DQN training started with run ID: {dqn_response.run_id}")
            print("  (Training in background...)")
        
        print("\n=== Demo Complete ===")
        print("Multiple training runs are now running in the background.")
        print("You can query their status using GetTrainingStatus with their run IDs.")
        
    except grpc.RpcError as e:
        print(f"\n✗ gRPC Error: {e.code()}: {e.details()}")
        print("\nMake sure the server is running: python server.py")
    
    finally:
        channel.close()


if __name__ == "__main__":
    main()
