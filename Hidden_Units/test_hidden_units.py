import subprocess
import os
import yaml

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
target_directory = os.path.dirname(script_dir)

file = "Hidden_Units/random_config_{0}.yaml"
static_command = "mlagents-learn {0} --run-id={1} --env=3DBall_environment/UnityEnvironment --no-graphics --force"
name = 'max'
parametr_name = "eps"
n_count = 3

hidden_units_value  = [16, 32, 64, 128, 256, 512]
lambda_values = [0.85, 0.875, 0.9, 0.925, 0.95, 0.975]
eps_values = []


def change_batch(yaml_list, hidden_units):
    yaml_list['behaviors']['3DBall']['network_settings']['hidden_units'] = hidden_units

def change_lambd(yaml_list, new_lambd):
    yaml_list['behaviors']['3DBall']['hyperparameters']['lambd'] = new_lambd

def change_eps(yaml_list, new_eps):
    yaml_list['behaviors']['3DBall']['hyperparameters']['epsilon'] = new_eps

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

config_value = 3

if __name__ == '__main__':
    os.chdir(target_directory)
    print(f"Current Working Directory changed to: {os.getcwd()}")
    for value in eps_values:
        edit_yaml(file.format(config_value), lambda doc: change_eps(doc, value))
        for i in range(n_count):
            standard_run(static_command.format(file.format(config_value), f"{name}-{parametr_name}{value}-config{config_value}-{i}"))