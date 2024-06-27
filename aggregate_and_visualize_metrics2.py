import json
import os
from collections import defaultdict
from visualizers import visualize_aggregated_metrics

def load_parameters(params_file):
    with open(params_file, 'r') as f:
        return json.load(f)

def load_evaluation_metrics(experiment_dir):
    metrics = defaultdict(list)
    for model in os.listdir(experiment_dir):
        model_dir = os.path.join(experiment_dir, model)
        if os.path.isdir(model_dir):
            for subdir, _, files in os.walk(model_dir):
                for file in files:
                    if file == 'evaluation_metrics.json':
                        eval_path = os.path.join(subdir, file)
                        with open(eval_path, 'r') as f:
                            data = json.load(f)
                            config = data['config']
                            metric_values = data['power_spectrum_error']
                            key = (config['Network']['name'], config['Training']['sampling_method'], config['Network'].get('alphas', ''))
                            metrics[key].append(metric_values)
    return metrics

def aggregate_metrics(params, metrics):
    aggregated_metrics = defaultdict(dict)
    for param in params:
        model = param['model']
        sampling_method = param['sampling_method']
        alphas = param.get('alphas', '')
        if alphas is None:
            alphas = ''
        key = (model, sampling_method, alphas)
        if key in metrics:
            aggregated_metrics[key] = {
                'mean': [sum(metric) / len(metric) for metric in zip(*metrics[key])],
                'std': [((sum((x - sum(metric) / len(metric)) ** 2 for x in metric) / len(metric)) ** 0.5) for metric in zip(*metrics[key])]
            }
    return aggregated_metrics

def save_aggregated_metrics(aggregated_metrics, output_file):
    # Convert tuple keys to string
    aggregated_metrics_str_keys = {str(key): value for key, value in aggregated_metrics.items()}
    with open(output_file, 'w') as f:
        json.dump(aggregated_metrics_str_keys, f, indent=4)

if __name__ == "__main__":
    experiment_dir = "saved_model/2024-05-15/h180_ep500_SampMeths_αs_0"  # Replace with the path to your experiment directory
    params_file_path = os.path.join(experiment_dir, "params_being_compared.json")
    output_file = os.path.join(experiment_dir, "aggregated_metrics.json")
    
    params = load_parameters(params_file_path)
    metrics = load_evaluation_metrics(experiment_dir)
    param_names = list(params[0].keys())
    aggregated_metrics = aggregate_metrics(params, metrics)
    aggregated_metrics['param_names'] = param_names  # Add param_names to aggregated_metrics
    save_aggregated_metrics(aggregated_metrics, output_file)
    
    print(f"Aggregated metrics saved to {output_file}")

    VISUALIZE = True
    if VISUALIZE:
        visualize_aggregated_metrics(aggregated_metrics, experiment_dir)
