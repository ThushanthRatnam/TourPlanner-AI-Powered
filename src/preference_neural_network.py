"""
Neural Network for Learning User Preferences
Uses PyTorch to build a deep learning model that learns user preferences
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple
import pandas as pd

class PreferenceNetwork(nn.Module):
    """
    Neural Network to learn and predict user preferences for places
    Architecture: Deep feedforward network with dropout and batch normalization
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32]):
        super(PreferenceNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers with batch normalization and dropout
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim
        
        # Output layer (preference score 0-1)
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class UserPreferenceLearner:
    """
    Learns user preferences using neural networks
    Combines user inputs with place features to predict preference scores
    """
    
    def __init__(self, input_dim: int, learning_rate: float = 0.001):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = PreferenceNetwork(input_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.input_dim = input_dim
        
    def create_user_profile(self, 
                           adventure_preferences: List[str],
                           budget: float,
                           num_days: int,
                           adventure_types_list: List[str]) -> np.ndarray:
        """
        Create a user profile vector from preferences
        """
        profile = []
        
        # Adventure type preferences (one-hot encoded)
        for adv_type in adventure_types_list:
            profile.append(1.0 if adv_type in adventure_preferences else 0.0)
        
        # Budget normalized (assuming max $1000 per day)
        profile.append(min(budget / 1000.0, 1.0))
        
        # Number of days normalized (assuming max 14 days)
        profile.append(min(num_days / 14.0, 1.0))
        
        return np.array(profile)
    
    def combine_features(self, 
                        user_profile: np.ndarray, 
                        place_features: np.ndarray) -> np.ndarray:
        """
        Combine user profile with place features to create input for neural network
        """
        # Concatenate user profile with place features
        combined = np.concatenate([user_profile, place_features])
        return combined
    
    def train_on_feedback(self, 
                         user_profiles: np.ndarray, 
                         place_features_list: np.ndarray,
                         ratings: np.ndarray,
                         epochs: int = 50):
        """
        Train the network on user feedback
        
        Args:
            user_profiles: Array of user profile vectors
            place_features_list: Array of place feature vectors
            ratings: Array of user ratings (0-1)
            epochs: Number of training epochs
        """
        self.model.train()
        
        # Combine features
        X = np.array([
            self.combine_features(user_profiles[i], place_features_list[i])
            for i in range(len(user_profiles))
        ])
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(ratings.reshape(-1, 1)).to(self.device)
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(X_tensor)
            loss = self.criterion(predictions, y_tensor)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    
    def predict_preference(self, 
                          user_profile: np.ndarray, 
                          place_features: np.ndarray) -> float:
        """
        Predict user's preference score for a place
        
        Returns:
            Preference score (0-1)
        """
        self.model.eval()
        
        with torch.no_grad():
            combined = self.combine_features(user_profile, place_features)
            X_tensor = torch.FloatTensor(combined).unsqueeze(0).to(self.device)
            prediction = self.model(X_tensor)
            
        return prediction.item()
    
    def rank_places(self,
                   user_profile: np.ndarray,
                   places_features: List[np.ndarray],
                   place_names: List[str]) -> List[Tuple[str, float]]:
        """
        Rank all places based on user preferences
        
        Returns:
            List of (place_name, preference_score) sorted by score
        """
        self.model.eval()
        
        scores = []
        for i, place_features in enumerate(places_features):
            score = self.predict_preference(user_profile, place_features)
            scores.append((place_names[i], score))
        
        # Sort by score (descending)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'input_dim': self.input_dim
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.input_dim = checkpoint['input_dim']


class ContentBasedRecommender:
    """
    Content-based recommendation system using cosine similarity
    Finds similar places based on features
    """
    
    def __init__(self):
        self.place_features = {}
        self.place_names = []
    
    def fit(self, place_names: List[str], place_features: List[np.ndarray]):
        """
        Fit the recommender with place features
        """
        self.place_names = place_names
        self.place_features = {name: features for name, features in zip(place_names, place_features)}
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def find_similar_places(self, 
                           place_name: str, 
                           top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find places similar to the given place
        
        Returns:
            List of (place_name, similarity_score)
        """
        if place_name not in self.place_features:
            return []
        
        target_features = self.place_features[place_name]
        similarities = []
        
        for name, features in self.place_features.items():
            if name != place_name:
                sim = self.cosine_similarity(target_features, features)
                similarities.append((name, sim))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def recommend_diverse_places(self,
                                user_preferences: List[str],
                                place_features_dict: Dict[str, np.ndarray],
                                top_k: int = 10) -> List[str]:
        """
        Recommend diverse places based on user preferences
        Uses diversity-aware recommendation
        """
        if not user_preferences:
            return list(place_features_dict.keys())[:top_k]
        
        recommendations = []
        remaining_places = set(place_features_dict.keys())
        
        # Start with a seed place from preferences
        seed_place = user_preferences[0] if user_preferences[0] in place_features_dict else list(place_features_dict.keys())[0]
        recommendations.append(seed_place)
        remaining_places.remove(seed_place)
        
        # Iteratively add diverse places
        while len(recommendations) < top_k and remaining_places:
            best_place = None
            best_score = -1
            
            for place in remaining_places:
                # Calculate diversity score (minimum similarity with already selected places)
                min_similarity = min([
                    self.cosine_similarity(
                        place_features_dict[place],
                        place_features_dict[rec]
                    ) for rec in recommendations
                ])
                
                # Prefer places with lower similarity (more diverse)
                diversity_score = 1 - min_similarity
                
                if diversity_score > best_score:
                    best_score = diversity_score
                    best_place = place
            
            if best_place:
                recommendations.append(best_place)
                remaining_places.remove(best_place)
            else:
                break
        
        return recommendations
