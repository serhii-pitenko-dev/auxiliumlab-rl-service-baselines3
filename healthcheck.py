"""
Health check client for the RL Training Service.

This script can be used to check if the gRPC server is healthy and accepting requests.
"""
import argparse
import logging
import sys
import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc

logger = logging.getLogger(__name__)


def check_health(host: str = "localhost", port: int = 50051, timeout: float = 5.0) -> bool:
    """
    Check the health of the gRPC server.
    
    Args:
        host: Server host address
        port: Server port
        timeout: Timeout in seconds for the health check
        
    Returns:
        True if server is healthy (SERVING), False otherwise
    """
    address = f"{host}:{port}"
    
    try:
        # Create a channel with a timeout
        with grpc.insecure_channel(address) as channel:
            # Wait for channel to be ready
            try:
                grpc.channel_ready_future(channel).result(timeout=timeout)
            except grpc.FutureTimeoutError:
                logger.error(f"Failed to connect to {address} within {timeout}s")
                return False
            
            # Create health stub
            health_stub = health_pb2_grpc.HealthStub(channel)
            
            # Make health check request
            request = health_pb2.HealthCheckRequest()
            response = health_stub.Check(request, timeout=timeout)  # type: ignore[attr-defined]
            
            # Check response status
            if response.status == health_pb2.HealthCheckResponse.SERVING:
                logger.info(f"✓ Server at {address} is SERVING")
                return True
            elif response.status == health_pb2.HealthCheckResponse.NOT_SERVING:
                logger.warning(f"✗ Server at {address} is NOT_SERVING")
                return False
            elif response.status == health_pb2.HealthCheckResponse.UNKNOWN:
                logger.warning(f"? Server at {address} status is UNKNOWN")
                return False
            else:
                logger.error(f"✗ Server at {address} returned unexpected status: {response.status}")
                return False
                
    except grpc.RpcError as e:
        logger.error(f"✗ gRPC error checking health at {address}: {e.code()} - {e.details()}")
        return False
    except Exception as e:
        logger.error(f"✗ Unexpected error checking health at {address}: {e}")
        return False


def main():
    """Main entry point for the health check client."""
    parser = argparse.ArgumentParser(
        description="Check the health of the RL Training Service gRPC server"
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Server host address (default: localhost)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=50051,
        help="Server port (default: 50051)"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Timeout in seconds (default: 5.0)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Perform health check
    is_healthy = check_health(args.host, args.port, args.timeout)
    
    # Exit with appropriate code
    if is_healthy:
        print(f"SUCCESS: Server is healthy at {args.host}:{args.port}")
        sys.exit(0)
    else:
        print(f"FAILURE: Server is not healthy at {args.host}:{args.port}")
        sys.exit(1)


if __name__ == "__main__":
    main()
