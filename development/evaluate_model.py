"""
Comprehensive Model Evaluation Script
Trains, evaluates, and explains the ML models with XAI techniques
"""
import sys
import os
# Add parent directory to path to import src modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import torch
import warnings
warnings.filterwarnings('ignore')

from src.data_loader import TripDataLoader
from src.preference_neural_network import PreferenceNetwork, UserPreferenceLearner
from src.model_evaluator import ModelEvaluator
from src.visualization import ModelVisualizer
from src.xai_explainer import XAIExplainer


def generate_synthetic_training_data(data_loader: TripDataLoader, 
                                     num_samples: int = 500) -> tuple:
    """
    Generate synthetic training data for model evaluation
    """
    print("\n" + "="*70)
    print("🔄 Generating Synthetic Training Data")
    print("="*70)
    
    adventure_types = ['Beach', 'Cultural', 'Wildlife', 'Hiking', 'Peace', 'Party', 'Urban', 'Adventure']
    
    X_data = []
    y_data = []
    
    for _ in range(num_samples):
        # Random user profile
        num_preferences = np.random.randint(1, 4)
        adventure_prefs = np.random.choice(adventure_types, num_preferences, replace=False).tolist()
        budget = np.random.uniform(100, 1500)
        num_days = np.random.randint(3, 15)
        
        # Create user profile
        learner = UserPreferenceLearner(input_dim=1)  # Dummy, just for profile creation
        user_profile = learner.create_user_profile(adventure_prefs, budget, num_days, adventure_types)
        
        # Random place
        place_idx = np.random.randint(0, len(data_loader.places_df))
        place_row = data_loader.places_df.iloc[place_idx]
        place_features = data_loader.get_feature_vector(place_row['POI Name'])
        
        # Combine features
        combined = np.concatenate([user_profile, place_features])
        X_data.append(combined)
        
        # Generate target (synthetic preference score)
        # Based on adventure type matching
        match_score = 0.0
        for adv_type in adventure_prefs:
            if place_row.get(f'adv_{adv_type}', 0) == 1:
                match_score += 1.0 / len(adventure_prefs)
        
        # Budget consideration
        place_fee = place_row.get('Fee (USD)', 0)
        budget_match = 1.0 - min(abs(place_fee - budget/num_days) / max(budget, 1), 1.0)
        
        preference_score = 0.7 * match_score + 0.3 * budget_match
        preference_score += np.random.normal(0, 0.1)  # Add noise
        preference_score = np.clip(preference_score, 0, 1)
        
        y_data.append(preference_score)
    
    X = np.array(X_data)
    y = np.array(y_data)
    
    print(f"✅ Generated {num_samples} training samples")
    print(f"   Feature dimension: {X.shape[1]}")
    print(f"   Target mean: {y.mean():.3f}, std: {y.std():.3f}")
    print("="*70 + "\n")
    
    return X, y


def get_feature_names(data_loader: TripDataLoader) -> list:
    """
    Generate feature names for the model
    """
    adventure_types = ['Beach', 'Cultural', 'Wildlife', 'Hiking', 'Peace', 'Party', 'Urban', 'Adventure']
    feature_names = []
    
    # User profile features
    for adv in adventure_types:
        feature_names.append(f'User_Pref_{adv}')
    feature_names.append('User_Budget_Normalized')
    feature_names.append('User_Days_Normalized')
    
    # Place features
    for adv in adventure_types:
        feature_names.append(f'Place_Has_{adv}')
    feature_names.append('Place_Fee')
    feature_names.append('Place_Duration')
    feature_names.append('Place_Weather_Score')
    feature_names.append('Place_Popularity')
    
    # District features (one-hot encoded - simplified)
    districts = data_loader.places_df['District'].unique()[:5]  # Top 5 districts
    for district in districts:
        feature_names.append(f'District_{district}')
    
    return feature_names[:len(feature_names)]  # Return actual length


def main():
    """
    Main evaluation pipeline
    """
    print("\n" + "🌴"*35)
    print("   SRI LANKA TRIP PLANNER - COMPREHENSIVE MODEL EVALUATION")
    print("🌴"*35 + "\n")
    
    # Initialize components
    print("📦 Loading data...")
    data_loader = TripDataLoader()
    
    # Generate training data
    X, y = generate_synthetic_training_data(data_loader, num_samples=1000)
    
    # Get feature names
    feature_names = get_feature_names(data_loader)[:X.shape[1]]
    
    # Initialize evaluator
    evaluator = ModelEvaluator(random_state=42)
    
    # ========================================================================
    # STEP 1: TRAIN/VALIDATION/TEST SPLIT
    # ========================================================================
    print("\n" + "="*70)
    print("📊 STEP 1: DATA SPLITTING")
    print("="*70)
    
    data_splits = evaluator.create_train_val_test_split(
        X, y,
        train_size=0.70,
        val_size=0.15,
        test_size=0.15
    )
    
    # ========================================================================
    # STEP 2: MODEL TRAINING WITH HYPERPARAMETERS
    # ========================================================================
    print("\n" + "="*70)
    print("🎓 STEP 2: MODEL TRAINING")
    print("="*70)
    
    # Hyperparameters
    hyperparameters = {
        'input_dim': X.shape[1],
        'hidden_dims': [128, 64, 32],
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 100,
        'dropout_rate': 0.3,
        'optimizer': 'Adam',
        'loss_function': 'MSE'
    }
    
    print("\n📋 Hyperparameters:")
    for key, value in hyperparameters.items():
        print(f"   {key:<20}: {value}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n💻 Device: {device}")
    
    # Initialize model
    model = PreferenceNetwork(
        input_dim=hyperparameters['input_dim'],
        hidden_dims=hyperparameters['hidden_dims']
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters['learning_rate'])
    criterion = torch.nn.MSELoss()
    
    # Train with validation
    training_history = evaluator.train_with_validation(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        data_splits=data_splits,
        device=device,
        epochs=hyperparameters['epochs'],
        batch_size=hyperparameters['batch_size'],
        early_stopping_patience=15,
        verbose=True
    )
    
    # ========================================================================
    # STEP 3: PERFORMANCE METRICS EVALUATION
    # ========================================================================
    print("\n" + "="*70)
    print("📈 STEP 3: PERFORMANCE METRICS")
    print("="*70)
    
    # Evaluate on all splits
    print("\nEvaluating on training set...")
    train_metrics = evaluator.evaluate_neural_network(
        model, data_splits['X_train'], data_splits['y_train'], device
    )
    
    print("Evaluating on validation set...")
    val_metrics = evaluator.evaluate_neural_network(
        model, data_splits['X_val'], data_splits['y_val'], device
    )
    
    print("Evaluating on test set...")
    test_metrics = evaluator.evaluate_neural_network(
        model, data_splits['X_test'], data_splits['y_test'], device
    )
    
    # Generate evaluation report
    report = evaluator.generate_evaluation_report(train_metrics, val_metrics, test_metrics)
    print(report)
    
    # ========================================================================
    # STEP 4: VISUALIZATIONS
    # ========================================================================
    print("\n" + "="*70)
    print("📊 STEP 4: CREATING VISUALIZATIONS")
    print("="*70)
    
    visualizer = ModelVisualizer(output_dir="evaluation_results")
    
    # Training history
    visualizer.plot_training_history(
        training_history['train_losses'],
        training_history['val_losses']
    )
    
    # Metrics comparison
    visualizer.plot_metrics_comparison(train_metrics, val_metrics, test_metrics)
    
    # Predictions vs actual (test set)
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(data_splits['X_test']).to(device)
        test_predictions = model(X_test_tensor).cpu().numpy().flatten()
    
    visualizer.plot_predictions_vs_actual(
        data_splits['y_test'],
        test_predictions,
        split_name="Test"
    )
    
    # Metrics table
    metrics_df = visualizer.create_metrics_table(train_metrics, val_metrics, test_metrics)
    print("\n📋 Metrics Table:")
    print(metrics_df)
    
    # Correlation matrix (sample of features)
    feature_df = pd.DataFrame(X[:500], columns=feature_names)
    feature_df['Target'] = y[:500]
    visualizer.plot_correlation_matrix(feature_df)
    
    # ========================================================================
    # STEP 5: EXPLAINABLE AI (XAI)
    # ========================================================================
    print("\n" + "="*70)
    print("🔍 STEP 5: EXPLAINABLE AI (XAI)")
    print("="*70)
    
    xai_explainer = XAIExplainer(model, device, output_dir="xai_results")
    
    # 5.1 Feature Importance (Permutation)
    feature_importance = xai_explainer.compute_feature_importance_permutation(
        data_splits['X_test'],
        data_splits['y_test'],
        feature_names,
        n_repeats=10
    )
    
    visualizer.plot_feature_importance(
        feature_names,
        np.array([feature_importance[fn] for fn in feature_names]),
        top_n=15
    )
    
    # 5.2 SHAP Analysis
    shap_results = xai_explainer.explain_with_shap(
        data_splits['X_train'],
        data_splits['X_test'],
        feature_names,
        max_samples=100
    )
    
    # 5.3 LIME Explanations
    lime_explanations = xai_explainer.explain_with_lime(
        data_splits['X_train'],
        data_splits['X_test'],
        feature_names,
        num_samples=3
    )
    
    # 5.4 Partial Dependence Plots
    xai_explainer.plot_partial_dependence(
        data_splits['X_test'],
        feature_names,
        feature_indices=list(range(min(9, len(feature_names))))
    )
    
    # Generate interpretation report
    interpretation = xai_explainer.generate_interpretation_report(
        feature_importance,
        shap_results
    )
    print(interpretation)
    
    # ========================================================================
    # STEP 6: SAVE RESULTS
    # ========================================================================
    print("\n" + "="*70)
    print("💾 STEP 6: SAVING RESULTS")
    print("="*70)
    
    # Save full report
    with open('evaluation_results/full_evaluation_report.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("SRI LANKA TRIP PLANNER - MODEL EVALUATION REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write("HYPERPARAMETERS:\n")
        f.write("-"*70 + "\n")
        for key, value in hyperparameters.items():
            f.write(f"{key:<20}: {value}\n")
        
        f.write(report)
        f.write(interpretation)
        
        f.write("\n\nFEATURE IMPORTANCE RANKING:\n")
        f.write("-"*70 + "\n")
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        for i, (feature, importance) in enumerate(sorted_features, 1):
            f.write(f"{i:2d}. {feature:<35} {importance:.6f}\n")
    
    print("✅ Saved full_evaluation_report.txt")
    
    # Save model
    torch.save(model.state_dict(), 'evaluation_results/best_model.pth')
    print("✅ Saved best_model.pth")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE!")
    print("="*70)
    print("\n📁 Results saved in:")
    print("   - evaluation_results/  (metrics, plots, tables)")
    print("   - xai_results/         (SHAP, LIME, PDP plots)")
    print("\n📊 Key Findings:")
    print(f"   - Test RMSE: {test_metrics['rmse']:.4f}")
    print(f"   - Test R²:   {test_metrics['r2']:.4f}")
    print(f"   - Test F1:   {test_metrics['f1_score']:.4f}")
    print(f"   - Most important feature: {sorted_features[0][0]}")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
