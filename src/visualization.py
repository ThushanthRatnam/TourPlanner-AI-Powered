"""
Visualization Module
Creates plots, graphs, and tables for model analysis
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import os

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class ModelVisualizer:
    """
    Create visualizations for model performance and analysis
    """
    
    def __init__(self, output_dir: str = "evaluation_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def plot_training_history(self,
                              train_losses: List[float],
                              val_losses: List[float],
                              save_path: str = None) -> None:
        """
        Plot training and validation loss curves
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        epochs = range(1, len(train_losses) + 1)
        
        ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        ax.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        
        # Mark best epoch
        best_epoch = np.argmin(val_losses) + 1
        best_val_loss = min(val_losses)
        ax.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, 
                   label=f'Best Epoch ({best_epoch})')
        ax.plot(best_epoch, best_val_loss, 'g*', markersize=15, 
                label=f'Best Val Loss: {best_val_loss:.4f}')
        
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss (MSE)', fontsize=12, fontweight='bold')
        ax.set_title('Training History: Loss Curves', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'training_history.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved training history plot: {save_path}")
        plt.close()
    
    def plot_metrics_comparison(self,
                               train_metrics: Dict[str, float],
                               val_metrics: Dict[str, float],
                               test_metrics: Dict[str, float],
                               save_path: str = None) -> None:
        """
        Create bar chart comparing metrics across splits
        """
        # Filter metrics to plot
        metrics_to_plot = ['rmse', 'mae', 'r2', 'accuracy', 'f1_score']
        available_metrics = [m for m in metrics_to_plot if m in train_metrics]
        
        if not available_metrics:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Regression metrics
        regression_metrics = [m for m in available_metrics if m in ['rmse', 'mae', 'r2', 'mape']]
        if regression_metrics:
            x = np.arange(len(regression_metrics))
            width = 0.25
            
            train_vals = [train_metrics[m] for m in regression_metrics]
            val_vals = [val_metrics[m] for m in regression_metrics]
            test_vals = [test_metrics[m] for m in regression_metrics]
            
            axes[0].bar(x - width, train_vals, width, label='Train', alpha=0.8)
            axes[0].bar(x, val_vals, width, label='Validation', alpha=0.8)
            axes[0].bar(x + width, test_vals, width, label='Test', alpha=0.8)
            
            axes[0].set_xlabel('Metric', fontsize=12, fontweight='bold')
            axes[0].set_ylabel('Value', fontsize=12, fontweight='bold')
            axes[0].set_title('Regression Metrics', fontsize=13, fontweight='bold')
            axes[0].set_xticks(x)
            axes[0].set_xticklabels([m.upper() for m in regression_metrics])
            axes[0].legend()
            axes[0].grid(True, alpha=0.3, axis='y')
        
        # Classification metrics
        classification_metrics = [m for m in available_metrics 
                                 if m in ['accuracy', 'precision', 'recall', 'f1_score']]
        if classification_metrics:
            x = np.arange(len(classification_metrics))
            width = 0.25
            
            train_vals = [train_metrics[m] for m in classification_metrics]
            val_vals = [val_metrics[m] for m in classification_metrics]
            test_vals = [test_metrics[m] for m in classification_metrics]
            
            axes[1].bar(x - width, train_vals, width, label='Train', alpha=0.8)
            axes[1].bar(x, val_vals, width, label='Validation', alpha=0.8)
            axes[1].bar(x + width, test_vals, width, label='Test', alpha=0.8)
            
            axes[1].set_xlabel('Metric', fontsize=12, fontweight='bold')
            axes[1].set_ylabel('Value', fontsize=12, fontweight='bold')
            axes[1].set_title('Classification Metrics', fontsize=13, fontweight='bold')
            axes[1].set_xticks(x)
            axes[1].set_xticklabels([m.replace('_', ' ').title() for m in classification_metrics])
            axes[1].legend()
            axes[1].grid(True, alpha=0.3, axis='y')
            axes[1].set_ylim([0, 1.1])
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'metrics_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved metrics comparison plot: {save_path}")
        plt.close()
    
    def plot_predictions_vs_actual(self,
                                   y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   split_name: str = "Test",
                                   save_path: str = None) -> None:
        """
        Scatter plot of predictions vs actual values
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Scatter plot
        axes[0].scatter(y_true, y_pred, alpha=0.6, edgecolors='k', linewidth=0.5)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', 
                    linewidth=2, label='Perfect Prediction')
        
        axes[0].set_xlabel('Actual Values', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Predicted Values', fontsize=12, fontweight='bold')
        axes[0].set_title(f'{split_name} Set: Predictions vs Actual', 
                         fontsize=13, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Residual plot
        residuals = y_true - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.6, edgecolors='k', linewidth=0.5)
        axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
        
        axes[1].set_xlabel('Predicted Values', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Residuals', fontsize=12, fontweight='bold')
        axes[1].set_title(f'{split_name} Set: Residual Plot', 
                         fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, f'predictions_vs_actual_{split_name.lower()}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved predictions vs actual plot: {save_path}")
        plt.close()
    
    def plot_feature_importance(self,
                               feature_names: List[str],
                               importance_scores: np.ndarray,
                               top_n: int = 15,
                               save_path: str = None) -> None:
        """
        Plot feature importance scores
        """
        # Sort by importance
        indices = np.argsort(importance_scores)[-top_n:]
        
        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))
        
        ax.barh(range(len(indices)), importance_scores[indices], alpha=0.8)
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
        ax.set_title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'feature_importance.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved feature importance plot: {save_path}")
        plt.close()
    
    def plot_correlation_matrix(self,
                               data: pd.DataFrame,
                               save_path: str = None) -> None:
        """
        Plot correlation matrix heatmap
        """
        correlation = data.corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                   ax=ax)
        
        ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'correlation_matrix.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved correlation matrix plot: {save_path}")
        plt.close()
    
    def create_metrics_table(self,
                            train_metrics: Dict[str, float],
                            val_metrics: Dict[str, float],
                            test_metrics: Dict[str, float],
                            save_path: str = None) -> pd.DataFrame:
        """
        Create formatted metrics table
        """
        metrics_data = {
            'Training': train_metrics,
            'Validation': val_metrics,
            'Test': test_metrics
        }
        
        df = pd.DataFrame(metrics_data).T
        
        # Format numbers
        df = df.round(4)
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'metrics_table.csv')
        
        df.to_csv(save_path)
        print(f"✅ Saved metrics table: {save_path}")
        
        # Also save as formatted text
        text_path = save_path.replace('.csv', '.txt')
        with open(text_path, 'w') as f:
            f.write(df.to_string())
        print(f"✅ Saved metrics table (text): {text_path}")
        
        return df
    
    def plot_genetic_algorithm_evolution(self,
                                        generation_history: List[Dict[str, float]],
                                        save_path: str = None) -> None:
        """
        Plot GA evolution: best and average fitness over generations
        """
        if not generation_history:
            return
        
        generations = [entry['generation'] for entry in generation_history]
        best_fitness = [entry['best_fitness'] for entry in generation_history]
        avg_fitness = [entry['avg_fitness'] for entry in generation_history]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(generations, best_fitness, 'b-', linewidth=2, label='Best Fitness', marker='o', markersize=4)
        ax.plot(generations, avg_fitness, 'r-', linewidth=2, label='Average Fitness', marker='s', markersize=4)
        
        ax.fill_between(generations, avg_fitness, best_fitness, alpha=0.2)
        
        ax.set_xlabel('Generation', fontsize=12, fontweight='bold')
        ax.set_ylabel('Fitness Score', fontsize=12, fontweight='bold')
        ax.set_title('Genetic Algorithm Evolution', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Annotate convergence point if applicable
        if len(best_fitness) > 5:
            # Check for convergence (when best fitness stops improving significantly)
            for i in range(len(best_fitness) - 5):
                if abs(best_fitness[i+5] - best_fitness[i]) < 0.01:
                    ax.axvline(x=generations[i], color='g', linestyle='--', alpha=0.7,
                              label=f'Convergence (Gen {generations[i]})')
                    break
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'ga_evolution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved GA evolution plot: {save_path}")
        plt.close()
