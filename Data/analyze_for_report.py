import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re

# 1. Load your file
try:
    df = pd.read_csv('final_combined_data_L.csv')
    print(f"✅ Loaded file with {len(df)} rows.")
except:
    print("❌ Error: Could not find 'final_combined_data_L.csv'")
    exit()

# 2. The Cleaner Function (Removes 'GB', handles 'Unknown')
def clean_money_and_ram(val):
    val = str(val).lower()
    if 'unknown' in val or val == 'nan':
        return 16.0  # Assume standard 16GB if missing
    # Remove everything that is NOT a number or a dot
    clean = re.sub(r'[^\d.]', '', val)
    try:
        return float(clean)
    except:
        return 16.0

# Apply the cleaning
print("🧹 Cleaning data...")
if 'Total RAM' in df.columns:
    df['Total RAM'] = df['Total RAM'].apply(clean_money_and_ram)

# Ensure everything else is numeric
features = ['batch_size', 'learning_rate', 'buffer_size', 'num_epoch', 'epsilon', 'beta']
targets = ['TimeElapsed', 'CumulativeReward']
analysis_df = df[features + targets].apply(pd.to_numeric, errors='coerce').fillna(0)

# 3. TASK 2.3: STATISTICAL ANALYSIS
print("\n" + "="*40)
print("   REPORT DATA (Copy to Section 0.6.1)")
print("="*40)
corr = analysis_df.corr()

# Extract specific numbers for your text
time_corr = corr.loc['batch_size', 'TimeElapsed']
reward_corr = corr.loc['learning_rate', 'CumulativeReward']

print(f"-> Correlation [Batch Size vs Time]: {time_corr:.4f}")
print(f"   (Interpretation: If this is positive, larger batches take longer. If negative, they are faster.)")
print(f"\n-> Correlation [Learning Rate vs Reward]: {reward_corr:.4f}")
print(f"   (Interpretation: Does changing learning rate actually improve the score?)")

# 4. TASK 2.4: VISUALIZATION GENERATION
print("\n🎨 Generating 3 Images for your Report...")

try:
    # Plot 1: Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Figure 1: Hyperparameter Correlation Matrix")
    plt.tight_layout()
    plt.savefig("Fig1_Correlation_Matrix.png")
    print("✅ Saved 'Fig1_Correlation_Matrix.png'")

    # Plot 2: Scatter Plot (Time vs Reward)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='TimeElapsed', y='CumulativeReward', data=analysis_df, alpha=0.3)
    plt.title("Figure 2: Training Efficiency (Time vs Reward)")
    plt.xlabel("Training Time (Seconds)")
    plt.ylabel("Final Reward")
    plt.tight_layout()
    plt.savefig("Fig2_Time_vs_Reward.png")
    print("✅ Saved 'Fig2_Time_vs_Reward.png'")

    # Plot 3: Boxplot (Batch Size Impact)
    plt.figure(figsize=(8, 6))
    # Filter to only show common batch sizes to avoid messy graph
    common_batches = analysis_df[analysis_df['batch_size'].isin([32, 64, 128, 256, 512, 1024])]
    if len(common_batches) > 0:
        sns.boxplot(x='batch_size', y='TimeElapsed', data=common_batches)
        plt.title("Figure 3: Impact of Batch Size on Training Speed")
        plt.tight_layout()
        plt.savefig("Fig3_BatchSize_Impact.png")
        print("✅ Saved 'Fig3_BatchSize_Impact.png'")
    else:
        print("⚠️ Skipped Fig3 (No standard batch sizes found)")

except Exception as e:
    print(f"⚠️ Graphing Error: {e}")

print("\n🚀 DONE. Open the images and write your report!")