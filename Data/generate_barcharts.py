import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


try:
    df = pd.read_csv('final_combined_data_L.csv')
    print(f" Loaded {len(df)} rows.")
except FileNotFoundError:
    print(" Error: 'final_combined_data_L.csv' not found. Make sure it's in the same folder.")
    exit()


sns.set_theme(style="whitegrid")


plt.figure(figsize=(8, 6))

sns.barplot(x='num_layers', y='CumulativeReward', data=df, palette='viridis')

plt.title("Figure 4: Impact of Network Depth on Reward")
plt.xlabel("Number of Layers")
plt.ylabel("Average Reward")
plt.savefig("Fig4_BarChart_Layers.png")
print(" Saved 'Fig4_BarChart_Layers.png'")


plt.figure(figsize=(8, 6))


common_sizes = [64, 128, 256, 512]
subset = df[df['hidden_units'].isin(common_sizes)]

sns.barplot(x='hidden_units', y='CumulativeReward', data=subset, palette='magma')

plt.title("Figure 5: Impact of Network Width on Reward")
plt.xlabel("Hidden Units (Neurons)")
plt.ylabel("Average Reward")
plt.savefig("Fig5_BarChart_Units.png")
print(" Saved 'Fig5_BarChart_Units.png'")

print("\n Doc saved in folder for the new PNGs.")