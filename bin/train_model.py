#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 by Inria
Authoried by Xiaoyu BIE (xiaoyu.bie@inria.fr)
License agreement in LICENSE.txt
"""
import os
import sys
import uuid
import configparser


from dvae.utils import find_project_root, Options, merge_configs
from dvae.learning_algo import LearningAlgorithm

if __name__ == "__main__":
    root_dir = find_project_root(__file__)

    params = Options().get_params()
    merged_config = merge_configs(params["device_cfg"], params["cfg"])

    # Update paths in the merged configuration
    merged_config["User"]["data_dir"] = merged_config["Paths"]["data_dir"]
    merged_config["User"]["saved_root"] = merged_config["Paths"]["saved_root"]

    # Use SLURM job ID for the temporary config file name if available, otherwise use a UUID
    if params["job_id"] is not None:
        print(f"[Train] Using SLURM job ID: {params['job_id']}")
        slurm_job_id = params["job_id"]
    else:
        # use short random numeric id
        slurm_job_id = str(uuid.uuid4())[:8]
        print(f"[Train] Using UUID: {slurm_job_id}")

    merged_config["User"]["slurm_job_id"] = slurm_job_id

    merged_config_path = os.path.join(
        root_dir, "config", "temp", f"merged_config_{slurm_job_id}.ini"
    )

    # Save the merged configuration temporarily
    with open(merged_config_path, "w") as configfile:
        config = configparser.ConfigParser()
        for section, section_values in merged_config.items():
            config[section] = section_values
        config.write(configfile)

    # Update params to use the merged configuration
    params["cfg"] = merged_config_path

    if not params["ss"]:
        print(f"[Train] Config File: {params['cfg']}")
        learning_algo = LearningAlgorithm(params=params)
        learning_algo.train()
    else:
        print("[Train] sorry ss is gone")

    # Clean up the temporary config file
    if os.path.exists(merged_config_path):
        os.remove(merged_config_path)
