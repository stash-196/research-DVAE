import json
import os
import argparse
from collections import defaultdict
from visualizers import visualize_aggregated_metrics
import pandas as pd

def load_parameters(params_file):
    with open(params_file, 'r') as f:
        return json.load(f)

def load_evaluation_metrics(experiment_dir, param_names):
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
                            if len(param_names) == 3:
                                key = (config['Network']['name'], config['Training']['sampling_method'], config['Network'].get('alphas', ''))
                            elif len(param_names) == 2:
                                key = (config['Network']['name'], config['Training']['sampling_method'])
                            metrics[key].append(metric_values)
    return metrics

def aggregate_metrics(params, param_names, metrics):
    aggregated_metrics = defaultdict(dict)
    for param in params:
        key = ()
        for param_name in param_names:
            p = param[param_name]
            if p == None:
                p = ''
            key += (p,)
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


def transform_to_dataframe(aggregated_metrics):
    rows = []
    param_names = aggregated_metrics.pop('param_names', [])
    
    for key, values in aggregated_metrics.items():
        params = list(key)
        means = values['mean']
        stds = values['std']
        
        variables = ['y', 'z', 'x', f'{params[0]}_Teacher-Forced', f'{params[0]}_Auto']

        # Add rows for each variable
        for i, variable in enumerate(variables):
            row = [variable] + params[1:] + [means[i], stds[i]]
            rows.append(row)
    
    # Create a DataFrame
    columns = ['variable'] + param_names[1:] + ['mean', 'std']
    df = pd.DataFrame(rows, columns=columns)
    
    return df

import matplotlib.pyplot as plt
import seaborn as sns

def visualize_error_bar(df, output_dir):
    # Extract the necessary columns
    x = df.columns[0]
    sub_x = df.columns[1]
    means = df['mean']
    stds = df['std']
    
    # Create the bar plot with error bars
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x=x, y=means, hue=sub_x, data=df, ci=None)
    ax.errorbar(df[x], means, yerr=stds, fmt='none', c='black', capsize=5)
    
    # Customize the plot
    plt.xlabel(x)
    plt.ylabel('Mean value with standard deviation')
    plt.title('Mean values with error bars')
    plt.legend(title=sub_x)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # save the plot
    plt.savefig(os.path.join(output_dir, 'error_bar_plot.png'))
    plt.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Aggregate evaluation metrics.')
    parser.add_argument('--exp_dir', type=str, required=True, help='Path to the experiment directory')

    args = parser.parse_args()
    experiment_dir = args.exp_dir

    params_file_path = os.path.join(experiment_dir, "params_being_compared.json")

    # make output dir if it doesn't exist
    output_dir = os.path.join(experiment_dir, "aggregated_results")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, "aggregated_metrics.json")
    
    params = load_parameters(params_file_path)
    param_names = list(params[0].keys())

    metrics = load_evaluation_metrics(experiment_dir, param_names)
    
    aggregated_metrics = aggregate_metrics(params, param_names, metrics)
    aggregated_metrics['param_names'] = param_names  # Add param_names to aggregated_metrics
    save_aggregated_metrics(aggregated_metrics, output_file)
    
    df = transform_to_dataframe(aggregated_metrics)
    visualize_error_bar(df, output_dir)
    print(f"Aggregated metrics saved to {output_file}")

    # VISUALIZE = True
    # if VISUALIZE:
    #     visualize_aggregated_metrics(aggregated_metrics, output_dir)
