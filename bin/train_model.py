#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 by Inria
Authoried by Xiaoyu BIE (xiaoyu.bie@inria.fr)
License agreement in LICENSE.txt
"""
import os
import uuid
from dvae.utils import find_project_root, Options
from dvae.utils import load_device_paths
from dvae.learning_algo import LearningAlgorithm

if __name__ == "__main__":
    root_dir = find_project_root(__file__)

    # Parse command-line arguments
    params = Options().get_params()

    # Load device-specific paths
    device_config = load_device_paths(
        os.path.join(root_dir, "config", "device_paths.yaml")
    )
    params["device_config"] = device_config

    # Add SLURM job ID or UUID
    params["job_id"] = params["job_id"] if params["job_id"] else str(uuid.uuid4())[:8]

    print(f"[Train] Using ID: {params['job_id']}")

    # Initialize and train
    print(f"[Train] Config File: {params['cfg']}")
    learning_algo = LearningAlgorithm(params=params)
    learning_algo.train()
