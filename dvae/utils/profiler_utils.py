import torch.profiler
import os

def profile_execution(func):
    def wrapper(*args, **kwargs):
        log_dir = './logs'
        
        # Ensure the log directory exists
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        print(f"Profiler will write logs to: {log_dir}")

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            print("Starting profiler...")
            result = func(*args, profiler=prof, **kwargs)
            print("Profiling completed")
            print(f"Profiler logs have been written to: {log_dir}")
            print("Contents of logs directory after profiling:")
            print(os.listdir(log_dir))  # Added to check the contents of the log directory after profiling
        return result
    return wrapper
