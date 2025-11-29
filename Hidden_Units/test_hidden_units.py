import subprocess
import os
import yaml

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
target_directory = os.path.dirname(script_dir)

file = "UserSettings/random_config_{0}.yaml"
static_command = "mlagents-learn {0} --run-id={1} --env=3DBall_environment/UnityEnvironment --no-graphics --force"
name = 'max'
n_count = 3

hidden_units_value  = [16, 32, 64, 128, 256, 512]


def change_batch(yaml_list, hidden_units):
    yaml_list['behaviors']['3DBall']['network_settings']['hidden_units'] = hidden_units

def edit_yaml(file, func):
    with open(file) as f:
        list_doc = yaml.safe_load(f)

    func(list_doc)
    with open(file, "w") as f:
        yaml.dump(list_doc, f)



def standard_run(command):
    try:
        os.chdir(target_directory)
        print(f"Current Working Directory changed to: {os.getcwd()}")
        
        # Run command in the target directory
        result = subprocess.run(command, shell=True, cwd=target_directory)
        return result.returncode
        
    except FileNotFoundError:
        print(f"Error: Directory not found at {target_directory}")
        return -1

config_value = 2

if __name__ == '__main__':
    os.chdir(target_directory)
    print(f"Current Working Directory changed to: {os.getcwd()}")
    for value in hidden_units_value:
        edit_yaml(file.format(config_value), lambda doc: change_batch(doc, value))
        for i in range(n_count):
            standard_run(static_command.format(file.format(config_value), f"{name}-hiddenunits{value}-config{config_value}-{i}"))