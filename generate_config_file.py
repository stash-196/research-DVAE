from collections import defaultdict
import itertools
import os
import json


class DefaultDict(defaultdict):
    def __missing__(self, key):
        return self.default_factory()


def generate_config_file(base_template, output_dir, experiment_name, testing_keys, config):
    config['test_keys'] = keys_being_compared

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Convert list values to comma-separated strings
    for key, value in config.items():
        if isinstance(value, list):
            config[key] = ', '.join(map(str, value))

    with open(base_template, 'r') as file:
        content = file.read()

    # Replace the placeholders with the actual values
    content = content.format_map(DefaultDict(lambda: '', **config))

    # Generating a unique name for the config file based on parameters
    label_keys = [
        f"{key}-{config[key]}" for key in config if key in testing_keys]

    output_file_name = '_'.join(
        [experiment_name, config['tag'], *label_keys]).replace(' ', '')
    output_file = os.path.join(output_dir, f"cfg_{output_file_name}.ini")

    with open(output_file, 'w') as file:
        file.write(content)

    return output_file


def get_configurations_for_model(params):
    param_names = list(params.keys())
    combinations = list(itertools.product(*params.values()))
    return [dict(zip(param_names, values)) for values in combinations]


if __name__ == "__main__":

    experiment_name = "h64_ep20000_esp50_SampMeths_AllRNNs_ActivFunc_0"

    models = [
        "RNN",
        "VRNN",
        "MT_RNN",
        "MT_VRNN"
    ]

    # Change to dictionary of lists
    # Network
    x_dim = [1]
    dense_x = [100]
    z_dim = [9]
    dense_z = [[16, 32]]

    dim_rnn = [64]
    alphas = [[0.00490695, 0.02916397, 0.01453569]]  # , [0.1, 0.01, 0.00267]]
    activation = ['relu', 'tanh']

    # Training
    lr = [0.001]
    alpha_lr = [0.01]
    epochs = [20000]
    early_stop_patience = [50]
    save_frequency = [50]
    gradient_clip = [1]
    optimize_alphas = [True]
    sampling_method = ['ss',
                       'ptf', 'mtf',
                       'even_bursts'
                       ]
    sampling_ratio = [0.8]

    # DataFrame
    dataset_name = ['Lorenz63']
    s_dim = [1]
    shuffle = [True]
    batch_size = [128]
    num_workers = [8]
    sequence_len = [1000]
    val_indices = [0.8]
    observation_process = ['only_x']

    model_params = {
        "RNN": {
            # User
            "experiment_name": [experiment_name],
            # Network
            "name": ["RNN"],
            "tag": ["RNN"],
            "x_dim": x_dim,
            "dense_x": dense_x,
            "dim_rnn": dim_rnn,
            "activation": activation,
            # Training
            "lr": lr,
            "epochs": epochs,
            "early_stop_patience": early_stop_patience,
            "save_frequency": save_frequency,
            "gradient_clip": gradient_clip,
            "sampling_method": sampling_method,
            "sampling_ratio": sampling_ratio,
            # DataFrame
            "dataset_name": dataset_name,
            "s_dim": s_dim,
            "shuffle": shuffle,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "sequence_len": sequence_len,
            "val_indices": val_indices,
            "observation_process": observation_process,
        },
        "VRNN": {
            # Network
            "name": ["VRNN"],
            "tag": ["VRNN"],
            "z_dim": z_dim,
            "dense_z": dense_z,
        },
        "MT_RNN": {
            # Network
            "name": ["MT_RNN"],
            "tag": ["MT_RNN"],
            "alphas": alphas,
            # Training
            "alpha_lr": alpha_lr,
            "optimize_alphas": optimize_alphas,
        },
        "MT_VRNN": {
            # Network
            "name": ["MT_VRNN"],
            "tag": ["MT_VRNN"],
        },
    }

    # Copy all components of "RNN" to "VRNN", "MT_RNN", and "MT_VRNN"
    # except for the ones that are specific to each model
    for model_name, params in model_params.items():
        if model_name != "RNN":
            for key, value in model_params["RNN"].items():
                if key not in params:
                    model_params[model_name][key] = value
    # Copy all components of "VRNN" to "MT_VRNN"
    # except for the ones that are specific to each model
    for key, value in model_params["VRNN"].items():
        if key not in model_params["MT_VRNN"]:
            model_params["MT_VRNN"][key] = value
    # Copy all components of "MT_RNN" to "MT_VRNN"
    # except for the ones that are specific to each model
    for key, value in model_params["MT_RNN"].items():
        if key not in model_params["MT_VRNN"]:
            model_params["MT_VRNN"][key] = value

    # Get all the parameters
    all_params = model_params["MT_VRNN"]

    # Get keys for which the len of the values is greater than 1 in "MT_VRNN"
    keys_being_compared = [
        key for key, value in model_params["MT_VRNN"].items() if len(value) > 1]

    # exclude models that are not in models. Don't run this before getting `params_being`
    model_params = {key: value for key,
                    value in model_params.items() if key in models}

    base_template = "config/sinusoid/cfg_base_template.ini"
    output_dir = os.path.join("config/sinusoid/generated/", experiment_name)

    all_configs = {
        model_name: get_configurations_for_model(params)
        for model_name, params in model_params.items()
    }

    for model_name, configs in all_configs.items():
        for config in configs:
            generate_config_file(base_template, output_dir,
                                 experiment_name, keys_being_compared, config)

    # Save parameters in JSON format
    json_params = []
    for model_name, configs in all_configs.items():
        for config in configs:
            json_params.append({
                "model": model_name,
                **{key: config.get(key, None) for key in keys_being_compared}
            })

    with open(os.path.join(output_dir, "params_being_compared.json"), "w") as file:
        json.dump(json_params, file, indent=4)
