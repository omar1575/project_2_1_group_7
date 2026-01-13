import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load the Big Data File
try:
    df = pd.read_csv('final_combined_data_L.csv')
    print(f"✅ Loaded {len(df)} rows.")
except FileNotFoundError:
    print("❌ Error: 'final_combined_data_L.csv' not found. Make sure it's in the same folder.")
    exit()

# 2. Setup the Style
sns.set_theme(style="whitegrid")

# --- CHART 1: Network Depth (Layers) ---
# Question: "Does adding more layers make the agent smarter?"
plt.figure(figsize=(8, 6))
# A Bar Plot automatically calculates the MEAN (height) and Error Bars (line on top)
sns.barplot(x='num_layers', y='CumulativeReward', data=df, palette='viridis')

plt.title("Figure 4: Impact of Network Depth on Reward")
plt.xlabel("Number of Layers")
plt.ylabel("Average Reward")
plt.savefig("Fig4_BarChart_Layers.png")
print("✅ Saved 'Fig4_BarChart_Layers.png'")

# --- CHART 2: Network Width (Hidden Units) ---
# Question: "Is a bigger brain (more neurons) always better?"
plt.figure(figsize=(8, 6))

# Filter to common sizes to avoid messy graph
common_sizes = [64, 128, 256, 512]
subset = df[df['hidden_units'].isin(common_sizes)]

sns.barplot(x='hidden_units', y='CumulativeReward', data=subset, palette='magma')

plt.title("Figure 5: Impact of Network Width on Reward")
plt.xlabel("Hidden Units (Neurons)")
plt.ylabel("Average Reward")
plt.savefig("Fig5_BarChart_Units.png")
print("✅ Saved 'Fig5_BarChart_Units.png'")

print("\n🚀 DONE. Check your folder for the new PNGs.")