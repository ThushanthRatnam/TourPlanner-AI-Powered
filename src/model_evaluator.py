"""
Model Evaluation Module
Provides train/test split, performance metrics, and evaluation framework
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
import torch
import torch.nn as nn
from copy import deepcopy


class ModelEvaluator:
    """
    Comprehensive evaluation framework for ML models
    Handles train/validation/test splits and performance metrics
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.metrics_history = {
            'train': [],
            'validation': [],
            'test': []
        }
        self.best_model_state = None
        self.best_val_loss = float('inf')
        
    def create_train_val_test_split(self,
                                     X: np.ndarray,
                                     y: np.ndarray,
                                     train_size: float = 0.7,
                                     val_size: float = 0.15,
                                     test_size: float = 0.15) -> Dict[str, np.ndarray]:
        """
        Split data into train, validation, and test sets
        
        Args:
            X: Feature matrix
            y: Target vector
            train_size: Proportion for training (default 70%)
            val_size: Proportion for validation (default 15%)
            test_size: Proportion for testing (default 15%)
            
        Returns:
            Dictionary with split data
        """
        assert abs(train_size + val_size + test_size - 1.0) < 1e-6, \
            "Split proportions must sum to 1.0"
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=self.random_state,
            shuffle=True
        )
        
        # Second split: separate validation from training
        val_ratio = val_size / (train_size + val_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            random_state=self.random_state,
            shuffle=True
        )
        
        print(f"✅ Data Split Complete:")
        print(f"   Training:   {len(X_train)} samples ({train_size*100:.1f}%)")
        print(f"   Validation: {len(X_val)} samples ({val_size*100:.1f}%)")
        print(f"   Test:       {len(X_test)} samples ({test_size*100:.1f}%)")
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }
    
    def calculate_regression_metrics(self,
                                    y_true: np.ndarray,
                                    y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate regression performance metrics
        
        Returns:
            Dictionary with RMSE, MAE, R², and MAPE
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Mean Absolute Percentage Error (avoid division by zero)
        mask = y_true != 0
        if np.any(mask):
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = 0.0
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }
    
    def calculate_classification_metrics(self,
                                        y_true: np.ndarray,
                                        y_pred: np.ndarray,
                                        threshold: float = 0.5) -> Dict[str, float]:
        """
        Calculate classification performance metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted probabilities
            threshold: Classification threshold
            
        Returns:
            Dictionary with accuracy, precision, recall, F1
        """
        y_pred_binary = (y_pred >= threshold).astype(int)
        y_true_binary = (y_true >= threshold).astype(int)
        
        accuracy = accuracy_score(y_true_binary, y_pred_binary)
        
        # Handle cases with only one class
        try:
            precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
            recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
            f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        except:
            precision = recall = f1 = 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def evaluate_neural_network(self,
                               model: nn.Module,
                               X: np.ndarray,
                               y: np.ndarray,
                               device: torch.device,
                               batch_size: int = 32) -> Dict[str, float]:
        """
        Evaluate neural network on a dataset
        
        Returns:
            Combined regression and classification metrics
        """
        model.eval()
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch_X = torch.FloatTensor(X[i:i+batch_size]).to(device)
                batch_pred = model(batch_X).cpu().numpy()
                predictions.extend(batch_pred)
        
        predictions = np.array(predictions).flatten()
        
        # Calculate both types of metrics
        regression_metrics = self.calculate_regression_metrics(y, predictions)
        classification_metrics = self.calculate_classification_metrics(y, predictions)
        
        return {**regression_metrics, **classification_metrics}
    
    def train_with_validation(self,
                             model: nn.Module,
                             optimizer: torch.optim.Optimizer,
                             criterion: nn.Module,
                             data_splits: Dict[str, np.ndarray],
                             device: torch.device,
                             epochs: int = 100,
                             batch_size: int = 32,
                             early_stopping_patience: int = 10,
                             verbose: bool = True) -> Dict[str, Any]:
        """
        Train neural network with validation and early stopping
        
        Returns:
            Training history and best model state
        """
        X_train, y_train = data_splits['X_train'], data_splits['y_train']
        X_val, y_val = data_splits['X_val'], data_splits['y_val']
        
        train_losses = []
        val_losses = []
        val_metrics_history = []
        
        patience_counter = 0
        
        print(f"\n{'='*60}")
        print(f"🎓 Training Neural Network with Validation")
        print(f"{'='*60}")
        print(f"Epochs: {epochs} | Batch Size: {batch_size} | Early Stopping: {early_stopping_patience}")
        print(f"{'='*60}\n")
        
        for epoch in range(epochs):
            model.train()
            epoch_losses = []
            
            # Training loop
            indices = np.random.permutation(len(X_train))
            for i in range(0, len(X_train), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_X = torch.FloatTensor(X_train[batch_indices]).to(device)
                batch_y = torch.FloatTensor(y_train[batch_indices].reshape(-1, 1)).to(device)
                
                optimizer.zero_grad()
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_losses.append(loss.item())
            
            avg_train_loss = np.mean(epoch_losses)
            train_losses.append(avg_train_loss)
            
            # Validation
            model.eval()
            with torch.no_grad():
                X_val_tensor = torch.FloatTensor(X_val).to(device)
                y_val_tensor = torch.FloatTensor(y_val.reshape(-1, 1)).to(device)
                val_predictions = model(X_val_tensor)
                val_loss = criterion(val_predictions, y_val_tensor).item()
                val_losses.append(val_loss)
                
                # Calculate validation metrics
                val_metrics = self.evaluate_neural_network(
                    model, X_val, y_val, device, batch_size
                )
                val_metrics_history.append(val_metrics)
            
            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = deepcopy(model.state_dict())
                patience_counter = 0
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"✨ Epoch {epoch+1:3d} | "
                          f"Train Loss: {avg_train_loss:.4f} | "
                          f"Val Loss: {val_loss:.4f} | "
                          f"Val RMSE: {val_metrics['rmse']:.4f} | "
                          f"Val R²: {val_metrics['r2']:.4f} ⭐")
            else:
                patience_counter += 1
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"   Epoch {epoch+1:3d} | "
                          f"Train Loss: {avg_train_loss:.4f} | "
                          f"Val Loss: {val_loss:.4f} | "
                          f"Val RMSE: {val_metrics['rmse']:.4f} | "
                          f"Val R²: {val_metrics['r2']:.4f}")
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\n⏹️  Early stopping at epoch {epoch+1}")
                print(f"   Best validation loss: {self.best_val_loss:.4f}")
                break
        
        # Load best model
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            print(f"\n✅ Loaded best model from epoch {epoch+1-patience_counter}")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_metrics_history': val_metrics_history,
            'best_val_loss': self.best_val_loss,
            'final_epoch': epoch + 1
        }
    
    def generate_evaluation_report(self,
                                  train_metrics: Dict[str, float],
                                  val_metrics: Dict[str, float],
                                  test_metrics: Dict[str, float]) -> str:
        """
        Generate a comprehensive evaluation report
        
        Returns:
            Formatted report string
        """
        report = []
        report.append("\n" + "="*70)
        report.append("📊 MODEL EVALUATION REPORT")
        report.append("="*70)
        
        # Regression Metrics
        report.append("\n📈 REGRESSION METRICS")
        report.append("-"*70)
        report.append(f"{'Metric':<20} {'Training':>15} {'Validation':>15} {'Test':>15}")
        report.append("-"*70)
        
        for metric in ['rmse', 'mae', 'r2', 'mape']:
            if metric in train_metrics:
                report.append(f"{metric.upper():<20} "
                            f"{train_metrics[metric]:>15.4f} "
                            f"{val_metrics[metric]:>15.4f} "
                            f"{test_metrics[metric]:>15.4f}")
        
        # Classification Metrics
        report.append("\n🎯 CLASSIFICATION METRICS")
        report.append("-"*70)
        report.append(f"{'Metric':<20} {'Training':>15} {'Validation':>15} {'Test':>15}")
        report.append("-"*70)
        
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            if metric in train_metrics:
                report.append(f"{metric.replace('_', ' ').title():<20} "
                            f"{train_metrics[metric]:>15.4f} "
                            f"{val_metrics[metric]:>15.4f} "
                            f"{test_metrics[metric]:>15.4f}")
        
        report.append("="*70)
        
        # Interpretation
        report.append("\n💡 INTERPRETATION")
        report.append("-"*70)
        
        # Check for overfitting
        train_rmse = train_metrics.get('rmse', 0)
        test_rmse = test_metrics.get('rmse', 0)
        if train_rmse > 0:
            rmse_diff = abs(test_rmse - train_rmse) / train_rmse
            if rmse_diff < 0.1:
                report.append("✅ Model generalizes well (train/test RMSE difference < 10%)")
            elif rmse_diff < 0.2:
                report.append("⚠️  Slight overfitting (train/test RMSE difference 10-20%)")
            else:
                report.append("❌ Potential overfitting (train/test RMSE difference > 20%)")
        
        # Model performance assessment
        test_r2 = test_metrics.get('r2', 0)
        if test_r2 >= 0.8:
            report.append("✅ Excellent predictive performance (R² ≥ 0.8)")
        elif test_r2 >= 0.6:
            report.append("✅ Good predictive performance (R² ≥ 0.6)")
        elif test_r2 >= 0.4:
            report.append("⚠️  Moderate predictive performance (R² ≥ 0.4)")
        else:
            report.append("❌ Low predictive performance (R² < 0.4)")
        
        # Classification performance
        test_f1 = test_metrics.get('f1_score', 0)
        if test_f1 >= 0.8:
            report.append("✅ Strong classification performance (F1 ≥ 0.8)")
        elif test_f1 >= 0.6:
            report.append("✅ Good classification performance (F1 ≥ 0.6)")
        else:
            report.append("⚠️  Moderate classification performance (F1 < 0.6)")
        
        report.append("="*70 + "\n")
        
        return "\n".join(report)
