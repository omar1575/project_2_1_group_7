import pandas as pd

try:
    df = pd.read_csv('combined.csv')
    print(f" Loaded 'combined.csv' with {len(df)} rows.")
except:
    print(" Error: Could not find 'combined.csv'. Please download it from the repo or your previous step.")
    exit()


features = [
    'batch_size', 'buffer_size', 'learning_rate', 'beta', 'epsilon', 'lambda', 
    'num_epoch', 'hidden_units', 'num_layers', 'time_horizon', 'max_steps', 
    'gamma', 'normalize', 'CPU Cores', 'Total RAM'
]

targets_mapping = {
    'Time Elapsed/s': 'TimeElapsed',
    '3DBall.Environment.CumulativeReward.mean': 'CumulativeReward'
}


df = df.rename(columns=targets_mapping)

defaults = {
    'hidden_units': 128, 'num_layers': 2, 'gamma': 0.99, 'max_steps': 500000,
    'time_horizon': 64, 'batch_size': 64, 'buffer_size': 10240, 'learning_rate': 3e-4,
    'beta': 0.001, 'epsilon': 0.2, 'lambda': 0.99, 'num_epoch': 3, 'normalize': True,
    'CPU Cores': 8, 'Total RAM': 16
}

for col in features:
    if col not in df.columns:
        df[col] = defaults.get(col, 0)
    df[col] = df[col].fillna(defaults.get(col, 0))


X = df[features]
y = df[['TimeElapsed', 'CumulativeReward']]
master = pd.concat([X, y], axis=1)

X.to_csv('X_features.csv', index=False)
y.to_csv('y_targets.csv', index=False)
master.to_csv('final_combined_data_L.csv', index=False)

print("\n Generated 3 BIG files:")
print("   -> X_features.csv")
print("   -> y_targets.csv")
print("   -> final_combined_data_L.csv")