import itertools
import subprocess
import yaml
import time

CONFIG_PATH = "Config/training_config.yaml"
ENV_PATH = "3DBall_environment/UnityEnvironment.exe"
BEHAVIOR_NAME = "3DBall"

def get_range(param_name, is_float=False):
    print(f"\nEnter values for '{param_name}':")
    start = float(input("  Start value: ")) if is_float else int(input("  Start value: "))
    end = float(input("  End value: ")) if is_float else int(input("  End value: "))
    step = float(input("  Step size: ")) if is_float else int(input("  Step size: "))

    values = []
    current = start
    while current <= end + 1e-9:
        values.append(round(current, 6) if is_float else current)
        current += step

    return values

params = {
    "batch_size": get_range("batch_size"),
    "buffer_size": get_range("buffer_size"),
    "learning_rate": get_range("learning_rate", is_float=True),
    "beta": get_range("beta", is_float=True),
    "epsilon": get_range("epsilon", is_float=True),
    "lambd": get_range("lambd", is_float=True),
    "num_epoch": get_range("num_epoch"),
    "hidden_units": get_range("hidden_units"),
    "num_layers": get_range("num_layers"),
    "gamma": get_range("gamma", is_float=True),
    "time_horizon": get_range("time_horizon")
}

keys = list(params.keys())
values = list(params.values())

print("\nTotal runs:", len(list(itertools.product(*values))))

for combo in itertools.product(*values):
    combo_dict = dict(zip(keys, combo))

    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    behavior = config["behaviors"][BEHAVIOR_NAME]

    hyper = behavior["hyperparameters"]
    hyper["batch_size"] = combo_dict["batch_size"]
    hyper["buffer_size"] = combo_dict["buffer_size"]
    hyper["learning_rate"] = combo_dict["learning_rate"]
    hyper["beta"] = combo_dict["beta"]
    hyper["epsilon"] = combo_dict["epsilon"]
    hyper["lambd"] = combo_dict["lambd"]
    hyper["num_epoch"] = combo_dict["num_epoch"]
    
    network = behavior["network_settings"]
    network["hidden_units"] = combo_dict["hidden_units"]
    network["num_layers"] = combo_dict["num_layers"]
    
    behavior["reward_signals"]["extrinsic"]["gamma"] = combo_dict["gamma"]
    behavior["time_horizon"] = combo_dict["time_horizon"]

    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    run_id = "run_" + "_".join(f"{k}{v}" for k, v in combo_dict.items())

    print("\nRunning config:")
    for k, v in combo_dict.items():
        print(f"  {k}: {v}")

    cmd = [
        "mlagents-learn",
        CONFIG_PATH,
        f"--run-id={run_id}",
        "--time-scale=20",
        f"--env={ENV_PATH}",
        "--no-graphics"
    ]

    subprocess.run(cmd)
    time.sleep(2)