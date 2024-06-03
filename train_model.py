#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 by Inria
Authoried by Xiaoyu BIE (xiaoyu.bie@inria.fr)
License agreement in LICENSE.txt
"""
import sys
from dvae.utils import Options, merge_configs
from dvae.learning_algo import LearningAlgorithm
import configparser


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
        print('sorry ss is gone')
