from collections import defaultdict
import itertools
import os

class DefaultDict(defaultdict):
    def __missing__(self, key):
        return self.default_factory()


def generate_config_file(base_template, output_dir, **kwargs):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Convert list values to comma-separated strings
    for key, value in kwargs.items():
        if isinstance(value, list):
            kwargs[key] = ', '.join(map(str, value))

    with open(base_template, 'r') as file:
        content = file.read()

    # Replace the placeholders with the actual values
    content = content.format_map(DefaultDict(lambda: '', **kwargs))

    # Generating a unique name for the config file based on parameters
    output_file_name = '_'.join([f"{key}_{value}" for key, value in kwargs.items()]).replace(', ', '_')
    output_file = f"{output_dir}/cfg_{output_file_name}.ini"
    
    with open(output_file, 'w') as file:
        file.write(content)
    
    return output_file

def get_configurations_for_model(params):
    param_names = list(params.keys())
    combinations = list(itertools.product(*params.values()))
    return [dict(zip(param_names, values)) for values in combinations]

if __name__ == "__main__":
    base_template = "config/sinusoid/cfg_base_template.ini"
    output_dir = "config/sinusoid/generated4"

    x_dim = [1]
    z_dim = [3]
    dense_x = [128]  
    dense_z = [[16, 32]]
    dim_RNN = [3]
    sequence_len = [3750, 7500, 15000]
    epochs = [300]
    early_stop_patience = [30]
    alphas = [[1/75, 1/900, 1/21600]]
    gradient_clip = [1]

    model_params = {
        "RNN": {
            "name": ["RNN"],
            "tag": ["RNN"],
            "x_dim": x_dim,
            "dense_x": dense_x,
            "dim_RNN": dim_RNN,
            "epochs": epochs,
            "early_stop_patience": early_stop_patience,
            "gradient_clip": gradient_clip,
            "sequence_len": sequence_len,
            "observation_process": ['3dto1d', '3dto1d_w_noise'],
        },
        "VRNN": {
            "name": ["VRNN"],
            "tag": ["VRNN"],
            "x_dim": x_dim,
            "z_dim": z_dim,
            "dense_x": dense_x,
            "dense_z": dense_z,
            "dim_RNN": dim_RNN,
            "epochs": epochs,
            "early_stop_patience": early_stop_patience,
            "gradient_clip": gradient_clip,
            "sequence_len": sequence_len,
            "observation_process": ['3dto1d', '3dto1d_w_noise'],
        },
        "MT_RNN": {
            "name": ["MT_RNN"],
            "tag": ["MT_RNN"],
            "x_dim": x_dim,
            "dense_x": dense_x,
            "dim_RNN": dim_RNN,
            "epochs": epochs,
            "early_stop_patience": early_stop_patience,
            "gradient_clip": gradient_clip,
            "alphas": alphas,
            "sequence_len": sequence_len,
            "observation_process": ['3dto1d', '3dto1d_w_noise'],
        },
        "MT_VRNN": {
            "name": ["MT_VRNN"],
            "tag": ["MT_VRNN"],
            "x_dim": x_dim,
            "z_dim": z_dim,
            "dense_x": dense_x,
            "dense_z": dense_z,
            "dim_RNN": dim_RNN,
            "epochs": epochs,
            "early_stop_patience": early_stop_patience,
            "alphas": alphas,
            "sequence_len": sequence_len,
            "observation_process": ['3dto1d', '3dto1d_w_noise'],
        },
    }

    all_configs = {
        model_name: get_configurations_for_model(params) 
        for model_name, params in model_params.items()
    }

    for model_name, configs in all_configs.items():
        for config in configs:
            generate_config_file(base_template, output_dir, **config)
