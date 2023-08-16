import os
import configparser
import pandas as pd
import re

base_dir = "saved_model"
all_data = []

# Helper function to extract value using regex pattern and return default value if not found
def extract_value(pattern, text, group_num=1, default=None):
    match = re.search(pattern, text)
    return match.group(group_num) if match else default

# Iterate through directories
for dirname in os.listdir(base_dir):
    dir_path = os.path.join(base_dir, dirname)
    if os.path.isdir(dir_path):
        # Check for the model final epoch file
        if any(re.match(r'.+_final_epoch\d+\.pt', f) for f in os.listdir(dir_path)):
            config_path = os.path.join(dir_path, 'config.ini')
            log_path = os.path.join(dir_path, 'log.txt')
            
            # Extract config data
            config = configparser.ConfigParser()
            config.read(config_path)
            config_data = {}
            for section in config.sections():
                config_data.update(dict(config[section]))

            # Extract log data
            with open(log_path, 'r') as f:
                logs = f.readlines()
            
            def extract_data_from_log(epoch_line):
                epoch_data = {}
                match = re.search(r'Val => tot: (\d+\.\d+) recon (\d+\.\d+) KL (\d+\.\d+)', epoch_line)
                if match:
                    epoch_data['Val Tot'] = float(match.group(1))
                    epoch_data['Recon'] = float(match.group(2))
                    epoch_data['KL'] = float(match.group(3))
                return epoch_data

            zero_epoch_data = extract_data_from_log(logs[0])
            twenty_epoch_data = extract_data_from_log(logs[20])
            final_epoch_data = extract_data_from_log(logs[-1])

            row_data = {
                "Directory": dirname,
                **config_data,
                "Final Epoch": int(extract_value(r'Epoch: (\d+)', logs[-1], default=0)),
                "0th Epoch Val Tot": zero_epoch_data.get('Val Tot', None),
                "0th Epoch Recon": zero_epoch_data.get('Recon', None),
                "0th Epoch KL": zero_epoch_data.get('KL', None),
                "20th Epoch Val Tot": twenty_epoch_data.get('Val Tot', None),
                "20th Epoch Recon": twenty_epoch_data.get('Recon', None),
                "20th Epoch KL": twenty_epoch_data.get('KL', None),
                "Final Epoch Val Tot": final_epoch_data.get('Val Tot', None),
                "Final Epoch Recon": final_epoch_data.get('Recon', None),
                "Final Epoch KL": final_epoch_data.get('KL', None),
            }
            all_data.append(row_data)

df = pd.DataFrame(all_data)
df.to_csv('saved_model/summary.csv', index=False)
