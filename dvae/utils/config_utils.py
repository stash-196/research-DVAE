import argparse
import configparser

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
        # Job ID if on Slurm
        self.parser.add_argument('--job_id', type=str, default=None, help='job ID')


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
