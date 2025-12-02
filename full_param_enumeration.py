import itertools
import subprocess
import yaml
import time

CONFIG_PATH = "Config/training_config.yaml"
ENV_PATH = "3DBall_environment/UnityEnvironment.exe"

# ---------------------------
# PARAMETER RANGES (REDUCED)
# ---------------------------
params = {
    "batch_size": list(range(1, 129, 13))[:10],     # 10 iterations
    "buffer_size": list(range(100, 24001, 2400))[:10],  # 10 iterations
    "learning_rate": [round(0.0001 + i * 0.0001, 5) for i in range(5)],  # 5 iterations
    "beta": [round(0.001 + i * 0.0001, 5) for i in range(5)],             # 5 iterations
    "epsilon": [round(0.1 + i * 0.05, 2) for i in range(5)]               # 5 iterations
}

keys = list(params.keys())
values = list(params.values())

# ---------------------------
# Loop over all combinations
# ---------------------------
for combo in itertools.product(*values):
    combo_dict = dict(zip(keys, combo))

    # Load YAML config
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    behavior_name = "3DBall"  # fixed since we know the behavior name
    behavior = config["behaviors"][behavior_name]

    # Update hyperparameters
    hyper = behavior["hyperparameters"]
    hyper["batch_size"] = combo_dict["batch_size"]
    hyper["buffer_size"] = combo_dict["buffer_size"]
    hyper["learning_rate"] = combo_dict["learning_rate"]
    hyper["beta"] = combo_dict["beta"]
    hyper["epsilon"] = combo_dict["epsilon"]

    # Save modified YAML
    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(config, f)

    print("\nRunning config:", combo_dict)
    run_id = "run_" + "_".join([f"{k}{v}" for k, v in combo_dict.items()])

    # Training command
    cmd = [
        "mlagents-learn",
        CONFIG_PATH,
        f"--run-id={run_id}",
        "--time-scale=20",
        f"--env={ENV_PATH}",
        "--no-graphics"
    ]

    subprocess.run(cmd)
    time.sleep(1)
