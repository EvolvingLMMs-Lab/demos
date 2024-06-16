import subprocess
import torch
import time

def get_gpu_utilization():
    """
    Gets the current GPU utilization using nvidia-smi command.

    Returns:
    list: A list of GPU utilization percentages.
    """
    try:
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'])
        utilization = result.decode('utf-8').strip().split('\n')
        utilization = [int(x) for x in utilization]
        return utilization
    except subprocess.CalledProcessError as e:
        print("Error querying GPU utilization:", e)
        return [0] * torch.cuda.device_count()

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
        time.sleep(interval)
        
def keep_gpu_busy(interval=1, threshold=10):
    """
    Keeps the GPU utilization above a certain threshold by performing
    dummy computations periodically.

    Args:
    interval (int): Time interval (in seconds) between each dummy computation.
    threshold (int): GPU utilization threshold.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please ensure you have a GPU with CUDA support.")
    
    device_count = torch.cuda.device_count()
    
    while True:
        utilization = get_gpu_utilization()
        print(f"Current GPU utilization: {utilization}%")

        for i in range(device_count):
            if utilization[i] < threshold:
                # Perform a dummy computation to keep the GPU busy
                gpu_stress_test(3600, 30, 0.05, i)
        
        # Wait for the specified interval before the next computation
        time.sleep(interval)

if __name__ == "__main__":
    keep_gpu_busy(interval=1, threshold=10)