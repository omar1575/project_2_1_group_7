import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

from CatBoostRegressor import CatBoostModel
from MLPRegressor import MLPModel
from RandomForest import RandomForestModel
import preprocess

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)

class ModelEvaluator:
    def __init__(self, data_path: str = "Data/Data.csv", test_size: float = 0.2, random_state: int = 42):
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state

        self.models = {}
        self.results = {}
        self.predictions = {}

        self._load_data()

    def _load_data(self):
        print("Loading and preprocessing data...")
        data = preprocess.get_processed_data(self.data_path)

        target_cols = ['time_elapased_seconds', 'cumulative_reward_mean']
        self.X, self.y = preprocess.split_features_targets(data, target_cols)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state)

        print(f"Data loaded: {len(self.X_train)} training samples, {len(self.X_test)} testing samples.")
        print(f"Features: {self.X.shape[1]}, Targets: {self.y.shape[1]}")
    
    def add_model(self, name: str, model):
        self.models[name] = model

    def train_all_models(self):
        print("\n" + "="*80)
        print("Training all models...")
        print("="*80 + "\n")

        for name, model in self.models.items():
            print(f"\nTraining model: {name}...")
            try:
                model.fit(self.X_train, self.y_train)
                print(f"Model {name} trained successfully.")
            except Exception as e:
                print(f"Error training model {name}: {e}")

    def evaluate_all_models(self):
        print("\n" + "="*80)
        print("Evaluating all models...")
        print("="*80 + "\n")

        for name, model in self.models.items():
            print(f"\nEvaluating model: {name}...")
            try:
                y_pred = model.predict(self.X_test)
                self.predictions[name] = y_pred

                metrics = {}
                target_names = ['Time (seconds)', 'Reward']

                for i, target_name in enumerate(target_names):
                    y_true = self.y_test.iloc[:, i].values
                    y_pred_target = y_pred[:, i]

                    metrics[target_name] = {
                        'MSE': mean_squared_error(y_true, y_pred_target),
                        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred_target)),
                        'MAE': mean_absolute_error(y_true, y_pred_target),
                        'R2': r2_score(y_true, y_pred_target)
                    }
                
                self.results[name] = metrics
                print(f"{name} evaluated successfully.")
            except Exception as e:
                print(f"Error evaluating model {name}: {e}")
    
    def print_results(self):
        print("\n" + "="*80)
        print("Model Evaluation Results")
        print("="*80 + "\n")

        for name, metrics in self.results.items():
            print(f"\nResults for model: {name}")
            print("-"*80)

            for target_name, vals in metrics.items():
                print(f"\nTarget: {target_name}")
                for metric_name, value in vals.items():
                    print(f"    {metric_name}: {value:.4f}")

    def create_comparison_table(self) -> pd.DataFrame:
        rows = []
        
        for model_name, metrics in self.results.items():
            for target_name, target_metrics in metrics.items():
                row = {
                    'Model': model_name,
                    'Target': target_name,
                    **target_metrics
                }
                rows.append(row)

        return pd.DataFrame.from_records(rows)
    
    def plot_metric_comparison(self, save_path: str = None):
        df = self.create_comparison_table()

        metrics = ['MSE', 'RMSE', 'MAE', 'R2']
        fig, axes = plt.subplots(2, len(metrics), figsize=(15, 10))

        targets = df['Target'].unique()

        for i, target in enumerate(targets):
            target_df = df[df['Target'] == target]

            for j, metric in enumerate(metrics):
                ax = axes[i, j]

                values = target_df[metric].values
                models = target_df['Model'].values
                
                colors = sns.color_palette("husl", len(models))
                bars = ax.bar(range(len(models)), values, color=colors)
                
                ax.set_xticks(range(len(models)))
                ax.set_xticklabels(models, rotation=45, ha='right')
                ax.set_ylabel(metric)
                ax.set_title(f'{metric} - {target}')
                ax.grid(axis='y', alpha=0.3)
                
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}',
                           ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Metric comparison plot saved to {save_path}")
        
        return fig

    def plot_predictions_vs_actual(self, save_path: str = None):
            n_models = len(self.predictions)
            fig, axes = plt.subplots(n_models, 2, figsize=(14, 5*n_models))
            
            if n_models == 1:
                axes = axes.reshape(1, -1)
            
            target_names = ['Time (seconds)', 'Reward']
            
            for i, (model_name, y_pred) in enumerate(self.predictions.items()):
                for j, target_name in enumerate(target_names):
                    ax = axes[i, j]
                    
                    y_true = self.y_test.iloc[:, j].values
                    y_pred_col = y_pred[:, j]
                    
                    ax.scatter(y_true, y_pred_col, alpha=0.5, s=30)
                    
                    min_val = min(y_true.min(), y_pred_col.min())
                    max_val = max(y_true.max(), y_pred_col.max())
                    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
                    
                    ax.set_xlabel(f'Actual {target_name}')
                    ax.set_ylabel(f'Predicted {target_name}')
                    ax.set_title(f'{model_name} - {target_name}')
                    ax.legend()
                    ax.grid(alpha=0.3)
                    
                    r2 = self.results[model_name][target_name]['R2']
                    ax.text(0.05, 0.95, f'R² = {r2:.4f}', 
                        transform=ax.transAxes, 
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Predictions vs actual plot saved to {save_path}")
            
            return fig

    def plot_error_distribution(self, save_path: str = None):
            n_models = len(self.predictions)
            fig, axes = plt.subplots(n_models, 2, figsize=(14, 5*n_models))
            
            if n_models == 1:
                axes = axes.reshape(1, -1)
            
            target_names = ['Time (seconds)', 'Reward (points)']
            
            for i, (model_name, y_pred) in enumerate(self.predictions.items()):
                for j, target_name in enumerate(target_names):
                    ax = axes[i, j]
                    
                    y_true = self.y_test.iloc[:, j].values
                    y_pred_col = y_pred[:, j]
                    errors = y_true - y_pred_col
                    
                    ax.hist(errors, bins=30, alpha=0.7, edgecolor='black')
                    
                    ax.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
                    
                    ax.set_xlabel(f'Error (Actual - Predicted) [{target_name}]')
                    ax.set_ylabel('Frequency')
                    ax.set_title(f'{model_name} - Error Distribution - {target_name}')
                    ax.legend()
                    ax.grid(axis='y', alpha=0.3)
                    
                    mean_error = errors.mean()
                    std_error = errors.std()
                    ax.text(0.05, 0.95, f'Mean: {mean_error:.4f}\nStd: {std_error:.4f}', 
                        transform=ax.transAxes, 
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Error distribution plot saved to {save_path}")
            
            return fig
        
    def generate_summary_report(self, output_path: str = "model_comparison_report.txt"):
            with open(output_path, 'w') as f:
                f.write("="*80 + "\n")
                f.write("MODEL COMPARISON REPORT\n")
                f.write("="*80 + "\n\n")
                
                f.write(f"Data: {self.data_path}\n")
                f.write(f"Training samples: {len(self.X_train)}\n")
                f.write(f"Test samples: {len(self.X_test)}\n")
                f.write(f"Number of features: {self.X.shape[1]}\n")
                f.write(f"Random state: {self.random_state}\n\n")
                
                df = self.create_comparison_table()
                f.write("SUMMARY TABLE\n")
                f.write("-"*80 + "\n")
                f.write(df.to_string(index=False))
                f.write("\n\n")
                
                f.write("DETAILED RESULTS\n")
                f.write("-"*80 + "\n\n")
                
                for model_name, metrics in self.results.items():
                    f.write(f"{model_name}\n")
                    f.write("-" * 80 + "\n")
                    
                    for target_name, target_metrics in metrics.items():
                        f.write(f"\n  {target_name}:\n")
                        for metric_name, value in target_metrics.items():
                            f.write(f"    {metric_name:8s}: {value:10.4f}\n")
                    f.write("\n")
                
                f.write("\nBEST MODELS BY METRIC\n")
                f.write("-"*80 + "\n")
                
                for target in df['Target'].unique():
                    target_df = df[df['Target'] == target]
                    f.write(f"\n{target}:\n")
                    
                    best_rmse = target_df.loc[target_df['RMSE'].idxmin()]
                    f.write(f"  Best RMSE: {best_rmse['Model']} ({best_rmse['RMSE']:.4f})\n")
                    
                    best_mae = target_df.loc[target_df['MAE'].idxmin()]
                    f.write(f"  Best MAE:  {best_mae['Model']} ({best_mae['MAE']:.4f})\n")
                    
                    best_r2 = target_df.loc[target_df['R2'].idxmax()]
                    f.write(f"  Best R2:   {best_r2['Model']} ({best_r2['R2']:.4f})\n")
            
            print(f"\nSummary report saved to {output_path}")

def main():
    print("="*80)
    print("ML Model Comparison and Evaluation")
    print("="*80)

    evaluator = ModelEvaluator(data_path="Data/Data.csv")

    print("\nInitializing models...")
    evaluator.add_model("CatBoost", CatBoostModel(iterations=500, verbose=False))
    evaluator.add_model("Random Forest", RandomForestModel(n_estimators=100))
    evaluator.add_model("MLP Regressor", MLPModel(hidden_layer_sizes=(256, 128, 64)))

    evaluator.train_all_models()

    evaluator.evaluate_all_models()

    evaluator.print_results()

    print("\n" + "="*80)
    print("Comparison Table")
    print("="*80 + "\n")
    df = evaluator.create_comparison_table()
    print(df.to_string(index=False))

    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    evaluator.plot_metric_comparison(save_path="metric_comparison.png")
    evaluator.plot_predictions_vs_actual(save_path="predictions_vs_actual.png")
    evaluator.plot_error_distribution(save_path="error_distribution.png")
    
    evaluator.generate_summary_report()

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - metric_comparison.png")
    print("  - predictions_vs_actual.png")
    print("  - error_distribution.png")
    print("  - model_comparison_report.txt")


if __name__ == "__main__":
    main()
