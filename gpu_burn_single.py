import torch
import time
import multiprocessing
import argparse

def gpu_stress_test(size, iterations, interval, gpu_id):
    """
    Function to perform a highly compute-intensive operation on a specific GPU.
    """
    # print(f"Running on GPU: {gpu_id}")

    # Set the specified GPU device
    torch.cuda.set_device(gpu_id)

    # Create large tensors and perform multiple compute-intensive operations
    for _ in range(iterations):
        a = torch.randn(size, size, device=f'cuda:{gpu_id}')
        b = torch.randn(size, size, device=f'cuda:{gpu_id}')
        c = a @ b  # Matrix multiplication

        # Additional operations to increase load
        for _ in range(10):
            c = c @ b
            c = c.sin()  # trigonometric operation for added complexity
            c = c.cos()
            c = c.tan()
        time.sleep(interval)

    # print(f"Ending on GPU {gpu_id}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="GPU Stress Test")
    parser.add_argument('--size', type=int, default=1600, help='Size of the square matrices for multiplication')
    parser.add_argument('--iterations', type=int, default=1000000, help='Number of iterations to perform the computation')
    parser.add_argument('--interval', type=float, default=0.1, help='Interval (in seconds) between iterations')
    parser.add_argument('--gpu_id', type=int, default=0, help='List of GPU IDs to use')

    args = parser.parse_args()
    return args.size, args.iterations, args.interval, args.gpu_id

if __name__ == "__main__":
    size, iterations, interval, gpu_id = parse_arguments()
    gpu_stress_test(size, iterations, interval,gpu_id)
