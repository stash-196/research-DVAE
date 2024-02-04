import concurrent.futures
import os

def run_parallel_visualizations(visualize_func, task_params):
    """
    Execute visualization tasks in parallel.

    Args:
    - visualize_func: The visualization function to be executed in parallel.
    - task_params: A list of dictionaries, where each dictionary contains arguments for the visualize_func.
    """
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Map each task to the executor
        futures = [executor.submit(visualize_func, **params) for params in task_params]
        
        # Iterate over the futures as they complete
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()  # Retrieve the result to catch any exceptions
            except Exception as e:
                print(f"Task raised an exception: {e}")

