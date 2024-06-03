# profiler_utils.py

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
                # Exclude CUDA if it's not available
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
            prof.export_stacks("/tmp/profiler_stacks.txt", "self_cpu_time_total")
            print(f"Profiler logs have been written to: {log_dir}")
        return result
    return wrapper
