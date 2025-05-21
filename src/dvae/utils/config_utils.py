# config_utils.py
import argparse


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def _initial(self):
        self.parser.add_argument(
            "--cfg", type=str, required=True, help="experiment config path"
        )
        self.parser.add_argument(
            "--use_pretrain", action="store_true", help="use pretrain"
        )
        self.parser.add_argument(
            "--pretrain_dict", type=str, default=None, help="pretrained model dict"
        )
        self.parser.add_argument(
            "--reload", action="store_true", help="resume training"
        )
        self.parser.add_argument(
            "--model_dir", type=str, default=None, help="model directory"
        )
        self.parser.add_argument(
            "--job_id", type=str, default=None, help="SLURM job ID"
        )

    def get_params(self):
        self._initial()
        self.opt = self.parser.parse_args()
        return vars(self.opt)
