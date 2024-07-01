import json
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def load_aggregated_metrics(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def generate_primary_x_labels(param_name, unique_values):
    if param_name == 'model':
        return [f"{value} TF" for value in unique_values] + [f"{value} Auto" for value in unique_values]
    elif param_name == 'sampling_method':
        return [f"{value} TF" for value in unique_values] + [f"{value} Auto" for value in unique_values]
    else:
        return ["TF", "Auto"]

def visualize_aggregated_metrics(aggregated_metrics, save_dir):
    # Extract parameter names
    param_names = aggregated_metrics.pop('param_names', [])
    
    # Variables to plot
    variables = ["y", "z"]
    
    # Group metrics by the first and second parameters for plotting
    grouped_metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for key_tuple, metrics in aggregated_metrics.items():
        primary_value, secondary_value, *residual_values = key_tuple
        grouped_metrics[primary_value][secondary_value][tuple(residual_values)].append((key_tuple, metrics))
    
    unique_primary_values = sorted(grouped_metrics.keys())
    primary_x_labels = generate_primary_x_labels(param_names[0], unique_primary_values)
    variables += primary_x_labels
    
    for primary_value, secondary_dict in grouped_metrics.items():
        for residual_values, sub_dict in secondary_dict.items():
            fig, ax = plt.subplots(figsize=(15, 8))
            title = f'Metrics for {param_names[0]} = {primary_value}'
            if residual_values:
                residual_str = ", ".join([f"{param_names[i+2]}={v}" for i, v in enumerate(residual_values) if i+2 < len(param_names)])
                title += f', {residual_str}'
            ax.set_title(title)
            ax.set_xlabel('Variables')
            ax.set_ylabel('Mean Power Spectrum Error')
            
            bar_width = 0.2
            index = np.arange(len(variables))

            for j, (secondary_value, entries) in enumerate(sub_dict.items()):
                mean_values = np.zeros(len(variables))
                std_values = np.zeros(len(variables))
                
                for (key, metrics) in entries:
                    # Fill in the mean and std values
                    for i, var in enumerate(metrics['mean']):
                        mean_values[i] = var
                        std_values[i] = metrics['std'][i]

                ax.bar(index + j * bar_width, mean_values[:len(index)], bar_width, yerr=std_values[:len(index)], label=f'{param_names[1]}={secondary_value}')

            ax.set_xticks(index + bar_width / 2 * (len(sub_dict) - 1))
            ax.set_xticklabels(variables, rotation=45, ha='right')
            ax.legend()

            plt.tight_layout()
            # Save plot considering the residual parameters
            residual_str = "_".join([f"{param_names[i+2]}={v}" for i, v in enumerate(residual_values) if i+2 < len(param_names)]) if residual_values else ""
            plt.savefig(os.path.join(save_dir, f'metrics_{primary_value}_{residual_str}.png'))
            plt.close()
            print(f"Saved plot for {param_names[0]} = {primary_value}, {residual_str}")

