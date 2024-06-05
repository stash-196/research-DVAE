from dvae.utils.profiler_utils import profile_execution
import torch
import os

@profile_execution
def simple_profiling(profiler=None):
    print("Simple profiling started")
    for _ in range(5):  # Simulate some workload
        x = torch.randn((100, 100))
        y = torch.mm(x, x)
        profiler.step()
    print("Simple profiling completed")

simple_profiling()

log_dir = './logs'
print(f"Profiler logs have been written to: {log_dir}")

# List files in the logs directory to confirm contents
print("Contents of logs directory:")
print(os.listdir(log_dir))
