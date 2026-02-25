"""
Test script to verify bidirectional gRPC communication.

This script tests:
1. Python → .NET: Calling the SimulationService
2. .NET → Python: (manual test - see GRPC_SETUP.md)
"""
import sys
import logging
from auxilium_rl.infra.external_env_adapter import GrpcExternalEnvAdapter

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_simulation_service():
    """Test Python calling .NET SimulationService."""
    logger.info("=" * 60)
    logger.info("Testing Python → .NET Communication")
    logger.info("=" * 60)
    
    try:
        # Create adapter (connects to .NET simulation)
        logger.info("Connecting to .NET simulation at localhost:50051...")
        adapter = GrpcExternalEnvAdapter("localhost:50051")
        logger.info("✓ Connected successfully!")
        
        # Test Reset
        logger.info("\nTesting Reset...")
        observation = adapter.reset(seed=42)
        logger.info(f"✓ Reset successful! Observation: {observation}")
        
        # Test Step
        logger.info("\nTesting Step...")
        obs, reward, terminated, truncated, info = adapter.step(action=1)
        logger.info(f"✓ Step successful!")
        logger.info(f"  Observation: {obs}")
        logger.info(f"  Reward: {reward}")
        logger.info(f"  Terminated: {terminated}")
        logger.info(f"  Truncated: {truncated}")
        logger.info(f"  Info: {info}")
        
        # Test another step
        logger.info("\nTesting another Step...")
        obs, reward, terminated, truncated, info = adapter.step(action=2)
        logger.info(f"✓ Step successful! Reward: {reward}")
        
        # Test Close
        logger.info("\nTesting Close...")
        adapter.close()
        logger.info("✓ Close successful!")
        
        logger.info("\n" + "=" * 60)
        logger.info("✓ All tests passed! Python → .NET communication works!")
        logger.info("=" * 60)
        return True
        
    except ConnectionError as e:
        logger.error(f"\n✗ Connection failed: {e}")
        logger.error("\nMake sure the .NET gRPC host is running:")
        logger.error("  cd AiSandBox.GrpcHost")
        logger.error("  dotnet run")
        return False
        
    except Exception as e:
        logger.error(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test entry point."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "gRPC Bidirectional Communication Test" + " " * 11 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    success = test_simulation_service()
    
    if success:
        print("\n✓ Communication test completed successfully!")
        print("\nNext steps:")
        print("1. ✓ Python can call .NET simulation (tested above)")
        print("2. ⏭ Test .NET calling Python training service:")
        print("   - Start Python training server: python server.py")
        print("   - Use PolicyTrainerClient in .NET code")
        print("\nSee GRPC_SETUP.md for detailed usage examples.")
        return 0
    else:
        print("\n✗ Communication test failed!")
        print("\nTroubleshooting:")
        print("1. Ensure .NET gRPC host is running:")
        print("   cd AiSandBox.GrpcHost")
        print("   dotnet run")
        print("2. Check that port 50051 is not blocked by firewall")
        print("3. Verify proto files are in sync")
        return 1


if __name__ == "__main__":
    sys.exit(main())
