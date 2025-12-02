import re
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Data/max.csv")


def extract_config(name):
    match = re.search(r'config(\d+)', name)
    return match.group(1) if match else None

filter = df["run_id"].str.startswith("max-eps")
df = df[filter]
df["config_id"] = df["run_id"].apply(extract_config)

configs = df["config_id"].unique()
colors = {cfg: plt.cm.tab10(i) for i, cfg in enumerate(configs)}

plt.figure(figsize=(8,5))
configs = df["config_id"].unique()
colors = {cfg: plt.cm.tab10(i) for i, cfg in enumerate(configs)}

for config, sub in df[df["config_id"]=='3'].groupby("config_id"):
    color = colors[config]

    hidden_vals = sorted(sub["epsilon"].unique())
    legend_label = f" Max Steps: {sub['max_steps'].iloc[0]}"
    means = []

    for value in hidden_vals:
        identical = sub[sub["epsilon"] == value]
        plt.scatter(
            identical["epsilon"],
            identical["Time Elapsed/s"],
            color=color,
            alpha=0.3,   
            s=35,            
        )
        mean_val = identical["Time Elapsed/s"].mean()
        means.append(mean_val)
        plt.plot(value, mean_val, marker="o", color=color)
    plt.plot(hidden_vals, means, color=color, label=legend_label)


plt.title("Epsilon vs Time")
plt.xlabel("Epsilon")
plt.ylabel("Time, s")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
