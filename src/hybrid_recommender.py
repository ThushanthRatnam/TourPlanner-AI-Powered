"""
Hybrid Recommendation System combining multiple techniques
Uses K-Means clustering for place grouping and content-based filtering
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple
import pandas as pd

class PlaceClusteringSystem:
    """
    Uses K-Means clustering to group similar places
    Helps in creating diverse itineraries
    """
    
    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.scaler = StandardScaler()
        self.place_to_cluster = {}
        self.cluster_centers = None
        
    def fit(self, place_names: List[str], place_features: List[np.ndarray]):
        """
        Fit the clustering model
        """
        features_array = np.array(place_features)
        
        # Standardize features
        features_scaled = self.scaler.fit_transform(features_array)
        
        # Fit K-Means
        self.kmeans.fit(features_scaled)
        
        # Store cluster assignments
        labels = self.kmeans.labels_
        for i, place_name in enumerate(place_names):
            self.place_to_cluster[place_name] = int(labels[i])
        
        self.cluster_centers = self.kmeans.cluster_centers_
        
    def get_cluster(self, place_name: str) -> int:
        """Get cluster ID for a place"""
        return self.place_to_cluster.get(place_name, -1)
    
    def get_places_from_cluster(self, cluster_id: int) -> List[str]:
        """Get all places from a specific cluster"""
        return [place for place, cid in self.place_to_cluster.items() if cid == cluster_id]
    
    def get_diverse_selection(self, 
                            place_names: List[str], 
                            num_places: int) -> List[str]:
        """
        Select diverse places ensuring representation from different clusters
        """
        if num_places >= len(place_names):
            return place_names
        
        # Group places by cluster
        cluster_groups = {}
        for place in place_names:
            cluster = self.get_cluster(place)
            if cluster not in cluster_groups:
                cluster_groups[cluster] = []
            cluster_groups[cluster].append(place)
        
        # Select places from each cluster proportionally
        selected = []
        places_per_cluster = max(1, num_places // len(cluster_groups))
        
        for cluster, places in cluster_groups.items():
            selected.extend(places[:places_per_cluster])
        
        # Fill remaining slots randomly
        remaining = [p for p in place_names if p not in selected]
        if len(selected) < num_places and remaining:
            selected.extend(remaining[:num_places - len(selected)])
        
        return selected[:num_places]


class HybridRecommendationSystem:
    """
    Hybrid recommendation system combining:
    1. Content-based filtering
    2. Collaborative-style scoring
    3. Cluster-based diversity
    """
    
    def __init__(self, n_clusters: int = 5):
        self.clustering = PlaceClusteringSystem(n_clusters)
        self.place_features = {}
        self.place_names = []
        self.popularity_scores = {}
        
    def fit(self, 
            place_names: List[str], 
            place_features: List[np.ndarray],
            popularity_data: Dict[str, float] = None):
        """
        Fit the recommendation system
        
        Args:
            place_names: List of place names
            place_features: List of feature vectors
            popularity_data: Optional dictionary of place popularity scores
        """
        self.place_names = place_names
        self.place_features = {name: features for name, features in zip(place_names, place_features)}
        
        # Set popularity scores (default to 0.5 if not provided)
        if popularity_data:
            self.popularity_scores = popularity_data
        else:
            self.popularity_scores = {name: 0.5 for name in place_names}
        
        # Fit clustering
        self.clustering.fit(place_names, place_features)
    
    def calculate_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def calculate_user_affinity(self, 
                               user_profile: np.ndarray,
                               place_features: np.ndarray) -> float:
        """
        Calculate how well a place matches user preferences
        """
        # Extract adventure type preferences from user profile and place features
        # Assuming first N elements are adventure type encodings
        n_adv_types = len(user_profile) - 2  # Subtract budget and days
        
        user_adv_prefs = user_profile[:n_adv_types]
        place_adv_types = place_features[:n_adv_types]
        
        # Calculate match score
        if np.sum(user_adv_prefs) > 0:
            adv_match = np.dot(user_adv_prefs, place_adv_types) / np.sum(user_adv_prefs)
        else:
            adv_match = 0.5
        
        return adv_match
    
    def hybrid_score(self,
                    place_name: str,
                    user_profile: np.ndarray,
                    neural_preference: float,
                    visited_places: List[str] = None,
                    weights: Dict[str, float] = None) -> float:
        """
        Calculate hybrid recommendation score
        
        Components:
        1. Neural network preference
        2. Content-based affinity
        3. Popularity score
        4. Diversity bonus (if visited places provided)
        """
        if weights is None:
            weights = {
                'neural': 0.4,
                'content': 0.3,
                'popularity': 0.2,
                'diversity': 0.1
            }
        
        if place_name not in self.place_features:
            return 0.0
        
        score = 0.0
        
        # 1. Neural preference
        score += weights['neural'] * neural_preference
        
        # 2. Content-based affinity
        place_features = self.place_features[place_name]
        affinity = self.calculate_user_affinity(user_profile, place_features)
        score += weights['content'] * affinity
        
        # 3. Popularity
        popularity = self.popularity_scores.get(place_name, 0.5)
        score += weights['popularity'] * popularity
        
        # 4. Diversity bonus
        if visited_places and weights['diversity'] > 0:
            place_cluster = self.clustering.get_cluster(place_name)
            visited_clusters = [self.clustering.get_cluster(p) for p in visited_places]
            
            # Bonus if from a different cluster
            if place_cluster not in visited_clusters:
                score += weights['diversity'] * 1.0
            else:
                score += weights['diversity'] * 0.3
        
        return score
    
    def recommend_next_place(self,
                           user_profile: np.ndarray,
                           neural_preference_scores: Dict[str, float],
                           visited_places: List[str],
                           available_places: List[str],
                           top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Recommend next places to visit
        
        Returns:
            List of (place_name, score) tuples
        """
        recommendations = []
        
        for place in available_places:
            if place not in visited_places:
                neural_score = neural_preference_scores.get(place, 0.5)
                hybrid_score = self.hybrid_score(
                    place, user_profile, neural_score, visited_places
                )
                recommendations.append((place, hybrid_score))
        
        # Sort by score
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations[:top_k]
    
    def create_balanced_itinerary(self,
                                 user_profile: np.ndarray,
                                 neural_preference_scores: Dict[str, float],
                                 num_places: int,
                                 available_places: List[str] = None) -> List[str]:
        """
        Create a balanced itinerary using greedy selection with diversity
        
        Returns:
            List of recommended place names
        """
        if available_places is None:
            available_places = self.place_names
        
        selected_places = []
        
        while len(selected_places) < num_places and len(selected_places) < len(available_places):
            recommendations = self.recommend_next_place(
                user_profile,
                neural_preference_scores,
                selected_places,
                available_places,
                top_k=3
            )
            
            if not recommendations:
                break
            
            # Add top recommendation
            selected_places.append(recommendations[0][0])
        
        return selected_places
    
    def get_cluster_summary(self) -> Dict[int, List[str]]:
        """
        Get summary of clusters
        
        Returns:
            Dictionary mapping cluster_id to list of places
        """
        summary = {}
        for place, cluster in self.clustering.place_to_cluster.items():
            if cluster not in summary:
                summary[cluster] = []
            summary[cluster].append(place)
        
        return summary
