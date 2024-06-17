import json
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def load_aggregated_metrics(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def visualize_aggregated_metrics(aggregated_metrics, save_dir):
    # Extract parameter keys from the first entry
    first_key = next(iter(aggregated_metrics))
    param_keys = first_key[:2]
    
    # If there are more than two parameters, use the rest for multiple plots
    extra_param_keys = first_key[2:] if len(first_key) > 2 else []

    for extra_key in extra_param_keys:
        # Group metrics by the extra key
        grouped_metrics = defaultdict(list)
        for key, metrics in aggregated_metrics.items():
            extra_value = key[2]
            grouped_metrics[extra_value].append((key[:2], metrics))
        
        for extra_value, entries in grouped_metrics.items():
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_title(f'Metrics for {extra_key} = {extra_value}')
            ax.set_xlabel(param_keys[0])
            ax.set_ylabel(param_keys[1])
            
            for (param1, param2), metrics in entries:
                mean_values = metrics['mean']
                std_values = metrics['std']
                x = np.arange(len(mean_values))
                
                ax.errorbar(x, mean_values, yerr=std_values, label=f'{param_keys[0]}={param1}, {param_keys[1]}={param2}')
            
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'metrics_{extra_key}_{extra_value}.png'))
            plt.close()
            print(f"Saved plot for {extra_key} = {extra_value}")

if __name__ == "__main__":
    aggregated_metrics_file = "path/to/aggregated_metrics.json"  # Replace with your path
    save_dir = "path/to/save/plots"  # Replace with your path
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    aggregated_metrics = load_aggregated_metrics(aggregated_metrics_file)
    visualize_aggregated_metrics(aggregated_metrics, save_dir)
    
    print(f"Plots saved to {save_dir}")
