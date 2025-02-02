import json
import os
import matplotlib.pyplot as plt
import numpy as np

def load_metrics(experiment_dir, verbose=False):
    metrics = {}
    reference_errors = None
    for model in os.listdir(experiment_dir):
        model_dir = os.path.join(experiment_dir, model)
        if os.path.isdir(model_dir):
            if verbose:
                print(f"Processing model directory: {model_dir}")
            for subdir, _, files in os.walk(model_dir):
                for file in files:
                    if file == 'evaluation_metrics.json':
                        eval_path = os.path.join(subdir, file)
                        if verbose:
                            print(f"Found evaluation_metrics.json: {eval_path}")
                        with open(eval_path, 'r') as f:
                            data = json.load(f)
                            config = data['config']
                            metric_values = data['power_spectrum_error']
                            if reference_errors is None:
                                reference_errors = metric_values[:2]  # y, z
                                if verbose:
                                    print(f"Set reference errors: {reference_errors}")
                            model_name = config['Network']['name']
                            sampling_method = config['Training']['sampling_method']
                            alphas = config['Network']['alphas'] if config['Network']['alphas'] != '' else None
                            key = (model_name, sampling_method, alphas)
                            if key not in metrics:
                                metrics[key] = []
                            metrics[key].append(metric_values[3:])  # teacher-forced, autonomous
                            if verbose:
                                print(f"Added metrics for key {key}: {metric_values[3:]}")
    return metrics, reference_errors

def plot_metrics(metrics, reference_errors, save_dir, verbose=False):
    training_methods = ["ss", "ptf", "mtf"]
    variables = ["y", "z", "RNN TF", "RNN Auto", "VRNN TF", "VRNN Auto", "MT_RNN TF", "MT_RNN Auto", "MT_VRNN TF", "MT_VRNN Auto"]
    model_indices = {
        "RNN": [2, 3],
        "VRNN": [4, 5],
        "MT_RNN": [6, 7],
        "MT_VRNN": [8, 9]
    }

    fig, axes = plt.subplots(1, len(training_methods), figsize=(20, 8), sharey=True)
    fig.suptitle('Comparison of Power Spectrum Error Across Models')

    for i, method in enumerate(training_methods):
        ax = axes[i]
        method_metrics = {var: [] for var in variables}
        alphas_set = set()

        # Initialize method_metrics with reference errors
        method_metrics["y"] = [reference_errors[0]]
        method_metrics["z"] = [reference_errors[1]]

        for key, values in metrics.items():
            model_name, sampling_method, alphas = key
            if sampling_method == method:
                if alphas:
                    alphas_set.add(alphas)
                indices = model_indices.get(model_name, [])
                if verbose:
                    print(f"Processing key {key} for method {method} with indices {indices}")
                for j, idx in enumerate(indices):
                    while len(method_metrics[variables[idx]]) < len(alphas_set):
                        method_metrics[variables[idx]].append(None)
                    method_metrics[variables[idx]].append(values[0][j])
                    if verbose:
                        print(f"Updated method_metrics for {variables[idx]}: {method_metrics[variables[idx]]}")

        alphas_list = sorted(list(alphas_set))
        bar_width = 0.2
        index = np.arange(len(variables))

        for j, alphas in enumerate(alphas_list):
            alpha_metrics = [
                method_metrics[var][j] if len(method_metrics[var]) > j and method_metrics[var][j] is not None else 0
                for var in variables
            ]
            ax.bar(index + j * bar_width, alpha_metrics, bar_width, label=f'alpha {alphas}')
            if verbose:
                print(f"Plotted bars for alphas {alphas}: {alpha_metrics}")

        # Add reference errors to the plot
        ax.bar(index[:2], [method_metrics["y"][0], method_metrics["z"][0]], bar_width, label='Reference Errors', color='orange')
        if verbose:
            print(f"Added reference errors to plot: {[method_metrics['y'][0], method_metrics['z'][0]]}")

        ax.set_xlabel('Variables')
        ax.set_title(f'Training Method: {method.upper()}')
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(variables, rotation=45, ha="right")

    plt.ylabel('Power Spectrum Error')
    plt.legend()
    plt.tight_layout()

    # Save the plot
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plot_path = os.path.join(save_dir, 'power_spectrum_error_comparison.png')
    plt.savefig(plot_path)
    plt.close()

    if verbose:
        print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    experiment_dir = "saved_model/2024-05-15/h180_ep500_SampMeths_αs_0"  # Replace with the path to your experiment directory
    
    metrics, reference_errors = load_metrics(experiment_dir, verbose=True)
    
    plot_metrics(metrics, reference_errors, experiment_dir, verbose=True)
    # Add calls to plot other metrics as needed
