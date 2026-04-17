from collections import defaultdict
import itertools
import os
import json


class DefaultDict(defaultdict):
    def __missing__(self, key):
        return self.default_factory()


def generate_config_file(
    base_template, output_dir, experiment_name, testing_keys, config
):
    config["test_keys"] = keys_being_compared

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Convert list values to comma-separated strings
    for key, value in config.items():
        if isinstance(value, list):
            config[key] = ", ".join(map(str, value))

    with open(base_template, "r") as file:
        content = file.read()

    # Replace the placeholders with the actual values
    content = content.format_map(DefaultDict(lambda: "", **config))

    # Generating a unique name for the config file based on parameters
    label_keys = [f"{key}-{config[key]}" for key in config if key in testing_keys]

    output_file_name = "_".join([experiment_name, config["tag"], *label_keys]).replace(
        " ", ""
    )
    output_file = os.path.join(output_dir, f"cfg_{output_file_name}.ini")

    with open(output_file, "w") as file:
        file.write(content)

    return output_file


def get_configurations_for_model(params):
    param_names = list(params.keys())
    combinations = list(itertools.product(*params.values()))
    return [dict(zip(param_names, values)) for values in combinations]


if __name__ == "__main__":

    # experiment_name = "ep20000_8alphas_esp50_nanBers_ptf_MT-RNN_SampRatios"
    # experiment_name = "20250902_" + "XHRO-01-11_coarse_all_power_alpha3d_ptf_seqlen1000_vary_MT-MTV"
    experiment_name = (
        "20260212_"
        # + "XHRO_len1000_drop0_ptf0.6-7-_clip1_AllLoss_LSTM_Subj70_ch1-2_hdi20s_ptientHigh"
        + "Lorenz_epoch10000_len1000_ptfAll_MissAll_clip1_LossNone_LSTM_hdi20_ptientHigh"
    )
    print("Experiment name:", experiment_name)

    models = [
        "RNN",
        # "VRNN",
        # "MT_RNN",
        # "MT_VRNN"
    ]
    rnn_types = [
        # "PLRNN",
        # "RNN",
        # "shPLRNN",
        "LSTM",
    ]

    # Change to dictionary of lists
    # Network
    x_dim = [1]
    dense_x = [1]
    z_dim = [9]
    dense_z = [[16, 32]]

    dim_rnn = [20]
    alphas = [
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        # [0.1],
        # [0.1, 0.1],
        # [0.1, 0.1, 0.1],
        # [ 0.09183, 0.64830, 0.73307, ],
        # [0.00490695, 0.02916397, 0.01453569], [0.1, 0.01, 0.00267],[0.1, 0.1, 0.1], [0.1], [0.01, 0.01], [0.9, 0.9]]
    ]
    activation = ["relu"]
    dropout_p = [
        0.0,
        # 0.1,
    ]

    # Training
    lr = [0.001]
    alpha_lr = [0.01]
    epochs = [10000]
    early_stop_patience = [200]
    save_frequency = [200]
    gradient_clip = [1.0]
    optimize_alphas = [True]
    sampling_method = [
        # "ss",
        "ptf",
        # 'mtf',
        # "sm"
    ]
    sampling_ratio = [
        0.0,
        # 0.01,
        # 0.05,
        0.1,
        # 0.2,
        0.3,
        # 0.4,
        0.5,
        # 0.6,
        0.7,
        0.8,
        # 0.9,
    ]
    auto_warm_start = [
        # 0.0,
        # 0.1,
        # 0.2,
        0.5,
    ]

    mask_autonomous_filled = [False]
    loss_mask_mode = [
        "none",
        # "weighted",
        # "strict",
    ]

    # DataFrame
    dataset_name = [
        # "Xhro",
        "Lorenz63",
        # "SHO",
        # "DampedSHO"
    ]
    if dataset_name[0] == "Lorenz63":
        dataset_label = [
            # "None",
            "sigma10_rho28_beta8d3_N108k_dt0.01",
            # "sigma10_rho24_beta8d3_N108k_dt0.01",
            # "sigma10_rho25_beta8d3_N108k_dt0.01",
            # "sigma10_rho26_beta8d3_N108k_dt0.01",
            # "sigma10_rho27_beta8d3_N108k_dt0.01",
            # "sigma10_rho30_beta8d3_N108k_dt0.01",
            # "sigma10_rho35_beta8d3_N108k_dt0.01",
            # "sigma10_rho40_beta8d3_N108k_dt0.01",
            # "sigma10_rho28_beta8d3_N108k_dt0.005",
            # "sigma10_rho28_beta8d3_N108k_dt0.02",
            # "sigma10_rho28_beta8d3_N108k_dt0.03",
            # "sigma10_rho28_beta8d3_N108k_dt0.04",
        ]
    elif dataset_name[0] == "Xhro":
        dataset_label = [
            # "None",
            # "XHRO_01_XH006",
            # "XHRO_04_XH057",
            "XHRO_02_XH070",
        ]
    elif dataset_name[0] == "SHO":
        dataset_label = [
            # "None",
            # "amp1,2,0.1_freq1,0.5,20_phas0,piD2,0_N10000_dt0.01",
            "amp1,2,0.5_freq1,0.5,20_phas0,piD2,0_N10000_dt0.01",
            # "amp1,2_freq1,0.5_phas0,piD2_N10000_dt0.01",
        ]
    elif dataset_name[0] == "DampedSHO":
        dataset_label = [
            # "None",
            # "omegas2pi,pi_gammas0.5,0.2_inst100_N1k_dt0.01",
            "omegas2pi_gammas0.5_inst100_N1k_dt0.01",
        ]
    else:
        # error
        raise ValueError(f"Unknown dataset_name: {dataset_name[0]}")

    if dataset_name[0] == "Xhro":
        mask_label = [
            "None",
        ]
    elif dataset_name[0] == "Lorenz63":
        mask_label = [
            "None",
            # "Markov_AvgLen15_0.0",
            "Markov_AvgLen15_0.1",
            # "Markov_AvgLen15_0.2",
            "Markov_AvgLen15_0.3",
            # "Markov_AvgLen15_0.4",
            "Markov_AvgLen15_0.5",
            # "Markov_AvgLen15_0.6",
            "Markov_AvgLen15_0.7",
            "Markov_AvgLen15_0.8",
            # "Markov_AvgLen15_0.9",
        ]
    else:
        mask_label = [
            "None",
        ]

    s_dim = [1]
    shuffle = [True]
    batch_size = [128]
    num_workers = [8]
    sequence_len = [1000]
    val_indices = [0.2]

    if dataset_name[0] == "Xhro":
        observation_process = [
            # "raw_all",
            "raw_ch1",
            "raw_ch2",
            "raw_ch3",
            # "raw_ch4",
            # "ch4_relative_powers",
            # "ch4_3_vars",
            # "ch4_alpha",
            # "all_ch_relative_powers",
            # "mixed_1d",
        ]
    elif dataset_name[0] == "Lorenz63":
        observation_process = [
            "only_x",
            # "only_x_w_noise",
            # "raw_noisy",
            # "relative_powers",
            # "3_vars",
            # "alpha",
        ]
    else:
        observation_process = [
            "raw",
            # "raw_noisy",
            # "relative_powers",
            # "3_vars",
            # "alpha",
        ]

    model_params = {
        "RNN": {
            # User
            "experiment_name": [experiment_name],
            # Network
            "name": ["RNN"],
            "tag": ["RNN"],
            "type_rnn": rnn_types,
            "dropout_p": dropout_p,
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
            "mask_autonomous_filled": mask_autonomous_filled,
            "auto_warm_start": auto_warm_start,
            "loss_mask_mode": loss_mask_mode,
            # DataFrame
            "dataset_name": dataset_name,
            "dataset_label": dataset_label,
            "mask_label": mask_label,
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
        key for key, value in model_params["MT_VRNN"].items() if len(value) > 1
    ]

    # exclude models that are not in models. Don't run this before getting `params_being`
    model_params = {key: value for key, value in model_params.items() if key in models}

    base_template = "config/general_signal/cfg_base_template.ini"
    output_dir = os.path.join("config/general_signal/generated/", experiment_name)

    all_configs = {
        model_name: get_configurations_for_model(params)
        for model_name, params in model_params.items()
    }

    for model_name, configs in all_configs.items():
        print(f"Number of configurations for {model_name}: {len(configs)}")
        for config in configs:
            generate_config_file(
                base_template, output_dir, experiment_name, keys_being_compared, config
            )

    # Save parameters in JSON format
    json_params = []
    for model_name, configs in all_configs.items():
        for config in configs:
            json_params.append(
                {
                    "model": model_name,
                    **{key: config.get(key, None) for key in keys_being_compared},
                }
            )

    with open(os.path.join(output_dir, "params_being_compared.json"), "w") as file:
        json.dump(json_params, file, indent=4)
