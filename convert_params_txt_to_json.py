import json

# The content of your params_being_compared.txt file
txt_content = """
RNN
{'sampling_method': '0-ss', 'alphas': '0-null'}
{'sampling_method': '0-ptf', 'alphas': '0-null'}
{'sampling_method': '0-mtf', 'alphas': '0-null'}
VRNN
{'sampling_method': '0-ss', 'alphas': '0-null'}
{'sampling_method': '0-ptf', 'alphas': '0-null'}
{'sampling_method': '0-mtf', 'alphas': '0-null'}
MT_RNN
{'sampling_method': '0-ss', 'alphas': '0-0.00490695, 0.02916397, 0.01453569'}
{'sampling_method': '1-ptf', 'alphas': '0-0.00490695, 0.02916397, 0.01453569'}
{'sampling_method': '2-mtf', 'alphas': '0-0.00490695, 0.02916397, 0.01453569'}
{'sampling_method': '0-ss', 'alphas': '1-0.1, 0.01, 0.00267'}
{'sampling_method': '1-ptf', 'alphas': '1-0.1, 0.01, 0.00267'}
{'sampling_method': '2-mtf', 'alphas': '1-0.1, 0.01, 0.00267'}
MT_VRNN
{'sampling_method': '0-ss', 'alphas': '0-0.00490695, 0.02916397, 0.01453569'}
{'sampling_method': '0-ss', 'alphas': '1-0.1, 0.01, 0.00267'}
{'sampling_method': '1-ptf', 'alphas': '0-0.00490695, 0.02916397, 0.01453569'}
{'sampling_method': '1-ptf', 'alphas': '1-0.1, 0.01, 0.00267'}
{'sampling_method': '2-mtf', 'alphas': '0-0.00490695, 0.02916397, 0.01453569'}
{'sampling_method': '2-mtf', 'alphas': '1-0.1, 0.01, 0.00267'}
"""


# Split the content into lines
lines = txt_content.split('\n')

# Initialize variables
params_list = []
current_model = None

# Helper function to remove the ids
def remove_ids(param_dict):
    new_dict = {}
    for key, value in param_dict.items():
        if '-' in value:
            new_dict[key] = value.split('-', 1)[1]
        else:
            new_dict[key] = value
    return new_dict

# Process each line
for line in lines:
    if line.strip() and not line.startswith('{'):
        current_model = line.strip()
    elif line.startswith('{'):
        param_dict = eval(line.strip())
        param_dict['model'] = current_model
        param_dict = remove_ids(param_dict)
        params_list.append(param_dict)

# Convert to JSON format
json_output = json.dumps(params_list, indent=4)

# Save the JSON output to a file
output_file = "params_being_compared.json"
with open(output_file, "w") as json_file:
    json_file.write(json_output)

print(f"Conversion complete. Saved as {output_file}")