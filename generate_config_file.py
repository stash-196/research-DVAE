from collections import defaultdict
import itertools

class DefaultDict(defaultdict):
    def __missing__(self, key):
        return self.default_factory()


def generate_config_file(base_template, output_dir, **kwargs):
    with open(base_template, 'r') as file:
        content = file.read()

    # Replace the placeholders with the actual values
    content = content.format_map(DefaultDict(lambda: '', **kwargs))


    # Generating a unique name for the config file based on parameters
    output_file_name = '_'.join([f"{key}_{value}" for key, value in kwargs.items()])
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
    output_dir = "config/sinusoid/generated"

    model_params = {
        "RNN": {
            "name": ["RNN"],
            "tag": ["RNN"],
            "x_dim": [75, 100],
            "dim_RNN": [3, 9],
            "observation_process": ['3dto1d', '3dto1d_w_noise'],
        },
        "VRNN": {
            "name": ["VRNN"],
            "tag": ["VRNN"],
            "z_dim": [1, 3],
            "x_dim": [75, 100],
            "dim_RNNs": [3, 9],
            "observation_process": ['3dto1d', '3dto1d_w_noise'],
        },
        "MT_RNN": {
            "name": ["MT_RNN"],
            "tag": ["MT_RNN"],
            "z_dim": [1, 3],
            "x_dim": [75, 100],
            "dim_RNN": [3, 9],
            "alphas": [[0.1, 0.5, 1.0]],
            "observation_process": ['3dto1d', '3dto1d_w_noise'],
        },
        "MT_VRNN": {
            "name": ["MT_VRNN"],
            "tag": ["MT_VRNN"],
            "z_dim": [1, 3],
            "x_dim": [75, 100],
            "dim_RNN": [3, 9],
            "observation_process": ['3dto1d', '3dto1d_w_noise'],
            "alphas": [[0.1, 0.5, 1.0]],
        },
    }

    all_configs = {
        model_name: get_configurations_for_model(params) 
        for model_name, params in model_params.items()
    }

    for model_name, configs in all_configs.items():
        for config in configs:
            generate_config_file(base_template, output_dir, **config)
