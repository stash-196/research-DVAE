#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 by Inria
Authoried by Xiaoyu BIE (xiaoyu.bie@inria.fr)
License agreement in LICENSE.txt
"""
import sys
import argparse
import configparser
from dvae.learning_algo import LearningAlgorithm
from dvae.learning_algo_ss import LearningAlgorithm_ss

class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def _initial(self):
        # Basic config file
        self.parser.add_argument('--cfg', type=str, default=None, help='config path')
        # Schedule sampling
        self.parser.add_argument('--ss', action='store_true', help='schedule sampling')
        self.parser.add_argument('--use_pretrain', action='store_true', help='if use pretrain')
        self.parser.add_argument('--pretrain_dict', type=str, default=None, help='pretrained model dict')
        # Resume training
        self.parser.add_argument('--reload', action='store_true', help='resume the training')
        self.parser.add_argument('--model_dir', type=str, default=None, help='model directory')
        # Device-specific config
        self.parser.add_argument('--device_cfg', type=str, default='config/cfg_device.ini', help='device config path')

    def get_params(self):
        self._initial()
        self.opt = self.parser.parse_args()
        params = vars(self.opt)
        return params

def merge_configs(device_config_path, experiment_config_path):
    config = configparser.ConfigParser()
    
    # Read device-specific settings
    config.read(device_config_path)
    device_config = config._sections['Paths']
    
    # Read experiment-specific settings
    config.read(experiment_config_path)
    experiment_config = {section: dict(config.items(section)) for section in config.sections()}
    
    # Merge configurations
    merged_config = {**experiment_config, 'Paths': device_config}
    
    return merged_config

if __name__ == '__main__':
    params = Options().get_params()
    merged_config = merge_configs(params['device_cfg'], params['cfg'])
    
    # Update paths in the merged configuration
    merged_config['User']['data_dir'] = merged_config['Paths']['data_dir']
    merged_config['User']['saved_root'] = merged_config['Paths']['saved_root']

    # Save the merged configuration temporarily
    merged_config_path = 'merged_config.ini'
    with open(merged_config_path, 'w') as configfile:
        config = configparser.ConfigParser()
        for section, section_values in merged_config.items():
            config[section] = section_values
        config.write(configfile)

    # Update params to use the merged configuration
    params['cfg'] = merged_config_path
    
    if not params['ss']:
        print(params['cfg'])
        learning_algo = LearningAlgorithm(params=params)
        learning_algo.train()
    else:
        learning_algo = LearningAlgorithm_ss(params=params)
        learning_algo.train()