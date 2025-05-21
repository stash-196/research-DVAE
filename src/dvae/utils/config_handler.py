# config_handler.py
import os
import socket
import yaml


def load_device_paths(config_path="config/device_paths.yaml"):
    """Load device-specific paths based on hostname."""
    hostname = socket.gethostname()
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Device config file not found: {config_path}")
    with open(config_path, "r") as f:
        device_configs = yaml.safe_load(f)["devices"]

    for device, config in device_configs.items():
        matcher = config["matcher"]
        if matcher["type"] == "hostname" and matcher["value"] == hostname:
            return {
                "saved_root": config["saved_root"],
                "data_dir": config["data_dir"],
                "device_name": device,
            }
        elif matcher["type"] == "substring" and matcher["value"] in hostname:
            return {
                "saved_root": config["saved_root"],
                "data_dir": config["data_dir"],
                "device_name": device,
            }

    raise ValueError(f"No device configuration found for hostname: {hostname}")
