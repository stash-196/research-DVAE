#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .path_manager import setup_project_root, find_project_root

# Automatically set up the project root when utils is imported
setup_project_root()


from .logger import get_logger, print_or_log
from .read_config import myconf
from .eval_metric import state_space_divergence, power_spectrum_error, prediction_error
from .loss import loss_ISD, loss_KLD, loss_JointNorm, loss_MPJPE, loss_MSE
from .model_mode_selector import create_autonomous_mode_selector
from .parallel_tasks_executers import run_parallel_visualizations
from .profiler_utils import profile_execution
from .config_utils import Options, merge_configs
