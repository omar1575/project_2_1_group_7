import subprocess
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
target_directory = os.path.dirname(script_dir)

static_command = "mlagents-learn Config/training_config.yaml --run-id={0} --env=3DBall_environment/UnityEnvironment --no-graphics --force"
name = 'max'
n_count = 5

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

if __name__ == '__main__':
    # Example: run a command
    for i in range(n_count):
        command = static_command.format(f'{name}-standart-'+str(i))
        standard_run(command)  # or any other command