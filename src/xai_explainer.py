"""
Explainable AI (XAI) Module
Implements SHAP, LIME, Feature Importance, and Partial Dependence Plots
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Callable
import torch
import torch.nn as nn
import os

try:
    import shap
    SHAP_AVAILABLE = True
except Exception as e:
    SHAP_AVAILABLE = False
    print(f"⚠️  SHAP not available: {str(e)}")

try:
    from lime import lime_tabular
    LIME_AVAILABLE = True
except Exception as e:
    LIME_AVAILABLE = False
    print(f"⚠️  LIME not available: {str(e)}")


class XAIExplainer:
    """
    Explainable AI methods for model interpretation
    """
    
    def __init__(self, model: nn.Module, device: torch.device, output_dir: str = "xai_results"):
        self.model = model
        self.device = device
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def predict_function(self, X: np.ndarray) -> np.ndarray:
        """
        Wrapper function for model predictions
        """
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy()
        return predictions.flatten()
    
    def compute_feature_importance_permutation(self,
                                              X_test: np.ndarray,
                                              y_test: np.ndarray,
                                              feature_names: List[str],
                                              n_repeats: int = 10) -> Dict[str, float]:
        """
        Compute feature importance using permutation method
        
        Args:
            X_test: Test features
            y_test: Test labels
            feature_names: List of feature names
            n_repeats: Number of times to permute each feature
            
        Returns:
            Dictionary with feature importance scores
        """
        print("\n" + "="*60)
        print("🔍 Computing Feature Importance (Permutation Method)")
        print("="*60)
        
        # Baseline performance
        baseline_pred = self.predict_function(X_test)
        baseline_mse = np.mean((y_test - baseline_pred) ** 2)
        
        importance_scores = {}
        
        for i, feature_name in enumerate(feature_names):
            mse_increases = []
            
            for _ in range(n_repeats):
                X_permuted = X_test.copy()
                # Permute the feature
                X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
                
                # Get predictions with permuted feature
                permuted_pred = self.predict_function(X_permuted)
                permuted_mse = np.mean((y_test - permuted_pred) ** 2)
                
                # Importance is the increase in error
                mse_increases.append(permuted_mse - baseline_mse)
            
            importance_scores[feature_name] = np.mean(mse_increases)
            print(f"  {feature_name:<30} Importance: {importance_scores[feature_name]:.6f}")
        
        # Normalize importance scores
        total_importance = sum(abs(v) for v in importance_scores.values())
        if total_importance > 0:
            importance_scores = {k: abs(v)/total_importance for k, v in importance_scores.items()}
        
        print("="*60 + "\n")
        return importance_scores
    
    def explain_with_shap(self,
                         X_train: np.ndarray,
                         X_test: np.ndarray,
                         feature_names: List[str],
                         max_samples: int = 100) -> Dict[str, Any]:
        """
        Generate SHAP explanations
        
        Args:
            X_train: Training data for background
            X_test: Test data to explain
            feature_names: List of feature names
            max_samples: Maximum samples for SHAP computation
            
        Returns:
            Dictionary with SHAP values and plots
        """
        if not SHAP_AVAILABLE:
            print("❌ SHAP not available. Skipping SHAP analysis.")
            return {}
        
        print("\n" + "="*60)
        print("🎯 SHAP (SHapley Additive exPlanations) Analysis")
        print("="*60)
        
        # Limit samples for computational efficiency
        X_train_sample = X_train[:min(max_samples, len(X_train))]
        X_test_sample = X_test[:min(max_samples, len(X_test))]
        
        # Create SHAP explainer
        explainer = shap.KernelExplainer(self.predict_function, X_train_sample)
        
        print(f"Computing SHAP values for {len(X_test_sample)} test samples...")
        shap_values = explainer.shap_values(X_test_sample)
        
        # Summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names, show=False)
        plt.title('SHAP Summary Plot', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'shap_summary.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved SHAP summary plot: {save_path}")
        plt.close()
        
        # Feature importance from SHAP
        shap_importance = np.abs(shap_values).mean(axis=0)
        shap_importance_dict = dict(zip(feature_names, shap_importance))
        
        # Bar plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names, 
                         plot_type="bar", show=False)
        plt.title('SHAP Feature Importance', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'shap_importance.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved SHAP importance plot: {save_path}")
        plt.close()
        
        print("="*60 + "\n")
        
        return {
            'shap_values': shap_values,
            'shap_importance': shap_importance_dict
        }
    
    def explain_with_lime(self,
                         X_train: np.ndarray,
                         X_test: np.ndarray,
                         feature_names: List[str],
                         num_samples: int = 5) -> List[Any]:
        """
        Generate LIME explanations for sample predictions
        
        Args:
            X_train: Training data
            X_test: Test data
            feature_names: List of feature names
            num_samples: Number of test samples to explain
            
        Returns:
            List of LIME explanations
        """
        if not LIME_AVAILABLE:
            print("❌ LIME not available. Skipping LIME analysis.")
            return []
        
        print("\n" + "="*60)
        print("🔬 LIME (Local Interpretable Model-agnostic Explanations)")
        print("="*60)
        
        # Create LIME explainer
        explainer = lime_tabular.LimeTabularExplainer(
            X_train,
            feature_names=feature_names,
            mode='regression',
            verbose=False
        )
        
        explanations = []
        
        for i in range(min(num_samples, len(X_test))):
            print(f"\nExplaining test sample {i+1}/{min(num_samples, len(X_test))}...")
            
            # Generate explanation
            exp = explainer.explain_instance(
                X_test[i],
                self.predict_function,
                num_features=len(feature_names)
            )
            
            explanations.append(exp)
            
            # Save explanation plot
            fig = exp.as_pyplot_figure()
            plt.title(f'LIME Explanation - Sample {i+1}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            save_path = os.path.join(self.output_dir, f'lime_explanation_sample_{i+1}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✅ Saved LIME explanation: {save_path}")
            plt.close()
            
            # Print top features
            print(f"\n  Top features for sample {i+1}:")
            for feature, weight in exp.as_list()[:5]:
                print(f"    {feature}: {weight:.4f}")
        
        print("\n" + "="*60 + "\n")
        
        return explanations
    
    def plot_partial_dependence(self,
                               X: np.ndarray,
                               feature_names: List[str],
                               feature_indices: List[int] = None,
                               grid_resolution: int = 50) -> None:
        """
        Create Partial Dependence Plots (PDP)
        
        Args:
            X: Feature matrix
            feature_names: List of feature names
            feature_indices: Indices of features to plot (default: all)
            grid_resolution: Number of points in the grid
        """
        print("\n" + "="*60)
        print("📊 Partial Dependence Plots (PDP)")
        print("="*60)
        
        if feature_indices is None:
            feature_indices = range(min(9, X.shape[1]))  # Plot up to 9 features
        
        n_features = len(feature_indices)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
        
        for idx, feature_idx in enumerate(feature_indices):
            print(f"Computing PDP for feature: {feature_names[feature_idx]}...")
            
            # Create grid for the feature
            feature_values = X[:, feature_idx]
            grid = np.linspace(feature_values.min(), feature_values.max(), grid_resolution)
            
            # Compute predictions for each grid point
            pdp_values = []
            
            for grid_value in grid:
                X_modified = X.copy()
                X_modified[:, feature_idx] = grid_value
                predictions = self.predict_function(X_modified)
                pdp_values.append(predictions.mean())
            
            # Plot
            axes[idx].plot(grid, pdp_values, linewidth=2, color='blue')
            axes[idx].fill_between(grid, pdp_values, alpha=0.3)
            axes[idx].set_xlabel(feature_names[feature_idx], fontsize=11, fontweight='bold')
            axes[idx].set_ylabel('Partial Dependence', fontsize=11, fontweight='bold')
            axes[idx].set_title(f'PDP: {feature_names[feature_idx]}', fontsize=12, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)
            
            # Add rug plot (distribution of actual values)
            axes[idx].plot(feature_values, 
                          np.ones_like(feature_values) * min(pdp_values), 
                          '|', color='red', alpha=0.1, markersize=10)
        
        # Hide extra subplots
        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'partial_dependence_plots.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Saved Partial Dependence Plots: {save_path}")
        plt.close()
        
        print("="*60 + "\n")
    
    def generate_interpretation_report(self,
                                      feature_importance: Dict[str, float],
                                      shap_results: Dict[str, Any] = None) -> str:
        """
        Generate comprehensive interpretation report
        
        Returns:
            Formatted interpretation text
        """
        report = []
        report.append("\n" + "="*70)
        report.append("🧠 MODEL INTERPRETATION REPORT")
        report.append("="*70)
        
        # Feature importance analysis
        report.append("\n📊 FEATURE IMPORTANCE ANALYSIS")
        report.append("-"*70)
        
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        report.append("\nTop 10 Most Influential Features:")
        for i, (feature, importance) in enumerate(sorted_features[:10], 1):
            bar_length = int(importance * 50)
            bar = "█" * bar_length
            report.append(f"  {i:2d}. {feature:<30} {importance:>6.4f} {bar}")
        
        # Interpretation
        report.append("\n💡 KEY INSIGHTS")
        report.append("-"*70)
        
        # Identify most important features
        top_feature = sorted_features[0][0]
        top_importance = sorted_features[0][1]
        
        report.append(f"\n1. PRIMARY DRIVER:")
        report.append(f"   '{top_feature}' is the most influential feature ({top_importance:.2%} importance)")
        report.append(f"   This suggests the model relies heavily on this feature for predictions.")
        
        # Check feature distribution
        top_5_importance = sum(imp for _, imp in sorted_features[:5])
        if top_5_importance > 0.8:
            report.append(f"\n2. FEATURE CONCENTRATION:")
            report.append(f"   Top 5 features account for {top_5_importance:.1%} of total importance")
            report.append(f"   The model shows strong feature concentration - predictions depend on few key features")
        else:
            report.append(f"\n2. FEATURE DISTRIBUTION:")
            report.append(f"   Top 5 features account for {top_5_importance:.1%} of total importance")
            report.append(f"   The model uses a diverse set of features - more balanced decision making")
        
        # SHAP insights if available
        if shap_results and 'shap_importance' in shap_results:
            report.append(f"\n3. SHAP ANALYSIS:")
            report.append(f"   SHAP values provide feature impact on individual predictions")
            report.append(f"   - Positive SHAP value: Feature increases prediction")
            report.append(f"   - Negative SHAP value: Feature decreases prediction")
            report.append(f"   See SHAP plots for detailed per-sample analysis")
        
        report.append("\n4. MODEL BEHAVIOR:")
        report.append("   The neural network has learned to:")
        report.append(f"   - Prioritize {sorted_features[0][0]} information")
        report.append(f"   - Consider {sorted_features[1][0]} as secondary factor")
        report.append(f"   - Balance multiple features for final predictions")
        
        report.append("\n5. DOMAIN ALIGNMENT:")
        report.append("   To validate if model behavior aligns with domain knowledge:")
        report.append("   - Check if top features match expected importance")
        report.append("   - Verify feature relationships make logical sense")
        report.append("   - Ensure no spurious correlations are driving decisions")
        
        report.append("\n6. RECOMMENDATIONS:")
        if top_importance > 0.5:
            report.append("   ⚠️  Consider collecting more diverse features")
            report.append("   ⚠️  Single feature dominance may indicate overfitting")
        else:
            report.append("   ✅ Feature importance distribution looks healthy")
            report.append("   ✅ Model considers multiple factors for decisions")
        
        report.append("\n" + "="*70 + "\n")
        
        return "\n".join(report)
