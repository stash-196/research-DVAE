import yaml

with open('environment.yml') as f:
    env = yaml.safe_load(f)

pip_packages = env['dependencies'][-1]['pip']
pip_command = f"/Users/stashtomonaga/opt/miniconda3/envs/research-DVAE/bin/pip install {' '.join(pip_packages)}"
print(pip_command)

# copy-paste the output in terminal