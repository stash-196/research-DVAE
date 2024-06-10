import torch.profiler
import os

def profile_execution(func):
    def wrapper(*args, **kwargs):
        log_dir = './logs'
        
        # Ensure the log directory exists
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        print(f"Profiler will write logs to: {log_dir}")

        try:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                ],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),  # Reduce profiling steps
                on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
                record_shapes=False,  # Disable if not necessary
                profile_memory=False,  # Disable if not necessary
                with_stack=False  # Disable if not necessary
            ) as prof:
                print("Starting profiler...")
                result = func(*args, profiler=prof, **kwargs)
                print("Profiling completed")
                print(f"Profiler logs have been written to: {log_dir}")
                print("Contents of logs directory after profiling:")
                print(os.listdir(log_dir))  # Check the contents of the log directory after profiling
            return result
        except Exception as e:
            print(f"An error occurred during profiling: {e}")
            raise
    return wrapper
