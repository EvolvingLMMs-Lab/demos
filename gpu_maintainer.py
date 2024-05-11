import subprocess
import time


def get_gpu_utilization():
    """Retrieve GPU utilization for all GPUs."""
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    )
    gpu_data = result.stdout.strip().split("\n")
    return {int(line.split(", ")[0]): int(line.split(", ")[1]) for line in gpu_data}


def monitor_and_run():
    processes = {}  # Dictionary to store subprocess PIDs for each GPU

    while True:
        gpu_utilization = get_gpu_utilization()

        print(f"====================== Detecting: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} ======================")
        for gpu_index, utilization in gpu_utilization.items():
            if utilization < 10 and gpu_index not in processes:
                # GPU is mostly idle, start gpu_burn.py if it's not already running
                print(f"Starting GPU-intensive process on GPU {gpu_index}...")
                proc = subprocess.Popen(
                    f"python /mnt/bn/vl-research/workspace/yhzhang/ml_envs/gpu_burn_single.py --iterations=24 --gpu_id {gpu_index}",
                    shell=True
                )
                processes[gpu_index] = proc
                time.sleep(0.01)
            elif utilization >= 10 and gpu_index in processes:
                # GPU utilization is high, terminate gpu_burn.py if it's running
                print(
                    f"Terminating GPU-intensive process on GPU {gpu_index} due to high utilization."
                )
                processes[gpu_index].terminate()
                del processes[gpu_index]
            elif utilization >= 10:
                print(f"GPU {gpu_index} is overutilized by user request.")
                
        print(f"====================== Finishing: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} ======================")
        time.sleep(5)  # Check every 10 seconds
        
        for gpu_index, utilization in gpu_utilization.items():
            if gpu_index in processes:
                try:
                    processes[gpu_index].terminate()
                except Exception as e:
                    print(f"Error terminating process on GPU {gpu_index}: {e}")
                del processes[gpu_index]


if __name__ == "__main__":
    monitor_and_run()
