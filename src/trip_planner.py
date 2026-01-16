"""
Main Sri Lanka Trip Planner
Orchestrates all ML models to create optimized trip itineraries
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import pandas as pd

from src.data_loader import TripDataLoader
from src.preference_neural_network import UserPreferenceLearner, ContentBasedRecommender
from src.genetic_optimizer import GeneticAlgorithmOptimizer
from src.hybrid_recommender import HybridRecommendationSystem


class SriLankaTripPlanner:
    """
    Main trip planner that combines:
    1. Neural Networks for preference learning
    2. Genetic Algorithm for optimization
    3. Hybrid recommendation system
    4. K-Means clustering for diversity
    """
    
    def __init__(self, data_directory: str = "."):
        """
        Initialize the trip planner
        
        Args:
            data_directory: Directory containing the CSV data files
        """
        print("🌴 Initializing Sri Lanka Trip Planner...")
        
        # Load data
        self.data_loader = TripDataLoader(data_directory)
        print(f"✅ Loaded {len(self.data_loader.places_df)} places")
        
        # Initialize neural network for preference learning
        self._setup_preference_learner()
        
        # Initialize hybrid recommender
        self._setup_recommender()
        
        # Initialize genetic algorithm optimizer
        self.ga_optimizer = GeneticAlgorithmOptimizer(
            population_size=100,
            mutation_rate=0.15,
            crossover_rate=0.7,
            elite_size=10,
            max_generations=80
        )
        
        print("✅ All models initialized successfully!")
    
    def _setup_preference_learner(self):
        """Setup the neural network preference learner"""
        # Calculate input dimension
        # User profile: adventure_types + budget + num_days
        # Place features: adventure_types + fee + time + category
        user_dim = len(self.data_loader.adventure_types_list) + 2
        place_dim = len(self.data_loader.adventure_types_list) + 3
        input_dim = user_dim + place_dim
        
        self.preference_learner = UserPreferenceLearner(
            input_dim=input_dim,
            learning_rate=0.001
        )
        
        # Initialize with some synthetic training data for cold start
        self._pretrain_preference_model()
    
    def _pretrain_preference_model(self):
        """
        Pretrain the model with synthetic data to avoid cold start
        """
        print("🔧 Pretraining preference model...")
        
        # Create synthetic training data
        num_samples = 200
        user_profiles = []
        place_features = []
        ratings = []
        
        places = self.data_loader.get_all_places()
        adventure_types = self.data_loader.adventure_types_list
        
        for _ in range(num_samples):
            # Random user preferences
            num_prefs = np.random.randint(1, 4)
            user_adv_prefs = np.random.choice(adventure_types, num_prefs, replace=False).tolist()
            user_budget = np.random.uniform(100, 1000)
            user_days = np.random.randint(3, 10)
            
            user_profile = self.preference_learner.create_user_profile(
                user_adv_prefs, user_budget, user_days, adventure_types
            )
            
            # Random place
            place = places.sample(1).iloc[0]
            place_feature = self.data_loader.get_feature_vector(place['POI Name'])
            
            # Synthetic rating based on match
            place_adv_types = place['adventure_type'].split(',')
            place_adv_types = [s.strip() for s in place_adv_types]
            
            match_score = len(set(user_adv_prefs) & set(place_adv_types)) / max(len(user_adv_prefs), 1)
            budget_match = 1.0 if place['Fee (USD)'] <= user_budget else 0.5
            rating = (match_score * 0.7 + budget_match * 0.3) + np.random.normal(0, 0.1)
            rating = np.clip(rating, 0, 1)
            
            user_profiles.append(user_profile)
            place_features.append(place_feature)
            ratings.append(rating)
        
        # Train the model
        self.preference_learner.train_on_feedback(
            np.array(user_profiles),
            np.array(place_features),
            np.array(ratings),
            epochs=30
        )
        
        print("✅ Preference model pretrained")
    
    def _setup_recommender(self):
        """Setup the hybrid recommendation system"""
        places = self.data_loader.get_all_places()
        place_names = places['POI Name'].tolist()
        place_features = [self.data_loader.get_feature_vector(name) for name in place_names]
        
        # Create popularity scores based on category (simplified)
        popularity = {}
        for name in place_names:
            place_info = self.data_loader.get_place_info(name)
            # Popular categories get higher scores
            if place_info['category'] in ['Cultural', 'Beach']:
                popularity[name] = 0.8
            elif place_info['category'] in ['Wildlife', 'Nature']:
                popularity[name] = 0.7
            else:
                popularity[name] = 0.6
        
        self.recommender = HybridRecommendationSystem(n_clusters=5)
        self.recommender.fit(place_names, place_features, popularity)
        
        print(f"✅ Hybrid recommender ready with {len(place_names)} places")
    
    def plan_trip(self,
                 adventure_types: List[str],
                 budget: float,
                 num_days: int,
                 travel_month: int = None,
                 optimization_level: str = 'balanced',
                 verbose: bool = True) -> Dict:
        """
        Plan a complete trip based on user preferences
        
        Args:
            adventure_types: List of preferred adventure types (e.g., ['Hiking', 'Cultural'])
            budget: Total budget in USD
            num_days: Number of days for the trip
            travel_month: Month of travel (1-12), defaults to current month
            optimization_level: 'fast', 'balanced', or 'thorough'
            verbose: Whether to print progress
        
        Returns:
            Dictionary with complete trip plan
        """
        if travel_month is None:
            travel_month = datetime.now().month
        
        if verbose:
            print("\n" + "="*60)
            print("🗺️  PLANNING YOUR SRI LANKA ADVENTURE")
            print("="*60)
            print(f"📅 Duration: {num_days} days")
            print(f"💰 Budget: ${budget:.2f}")
            print(f"🎯 Adventure Types: {', '.join(adventure_types)}")
            print(f"📆 Travel Month: {travel_month}")
            print("="*60 + "\n")
        
        # Step 1: Create user profile
        user_profile = self.preference_learner.create_user_profile(
            adventure_types, budget, num_days, self.data_loader.adventure_types_list
        )
        
        # Step 2: Filter places by preferences
        places_by_type = self.data_loader.get_places_by_adventure_type(adventure_types)
        places_by_budget = self.data_loader.get_places_by_budget(budget)
        
        # Find intersection using indices
        available_places = places_by_type[places_by_type['POI Name'].isin(places_by_budget['POI Name'])]
        
        if available_places.empty:
            # Relax constraints if no places found
            available_places = self.data_loader.get_places_by_budget(budget * 1.5)
        
        available_place_names = available_places['POI Name'].tolist()
        
        if verbose:
            print(f"🔍 Found {len(available_place_names)} places matching your preferences")
        
        # Step 3: Get neural network preference scores
        neural_scores = {}
        for place_name in available_place_names:
            place_features = self.data_loader.get_feature_vector(place_name)
            score = self.preference_learner.predict_preference(user_profile, place_features)
            neural_scores[place_name] = score
        
        # Step 4: Use hybrid recommender to create candidate list
        # Increase candidate pool based on trip duration
        candidate_pool_size = min(50 + (num_days * 5), len(available_place_names))
        candidate_places = self.recommender.create_balanced_itinerary(
            user_profile,
            neural_scores,
            num_places=candidate_pool_size,
            available_places=available_place_names
        )
        
        if verbose:
            print(f"🎯 Selected {len(candidate_places)} candidate places for optimization")
        
        # Step 5: Use Genetic Algorithm to optimize the itinerary
        if verbose:
            print(f"\n🧬 Running Genetic Algorithm optimization...")
        
        # Adjust GA parameters based on optimization level
        if optimization_level == 'fast':
            self.ga_optimizer.max_generations = 30
            self.ga_optimizer.population_size = 50
        elif optimization_level == 'thorough':
            self.ga_optimizer.max_generations = 150
            self.ga_optimizer.population_size = 150
        else:  # balanced
            self.ga_optimizer.max_generations = 80
            self.ga_optimizer.population_size = 100
        
        optimized_result = self.ga_optimizer.get_optimized_itinerary(
            candidate_places,
            num_days,
            self.data_loader,
            user_profile,
            self.preference_learner,
            budget,
            travel_month,
            verbose=verbose
        )
        
        # Step 6: Get additional recommendations
        top_recommendations = self.get_recommendations(
            adventure_types,
            budget,
            top_k=10
        )
        
        # Step 7: Compile final results
        if verbose:
            print("\n" + "="*60)
            print("✨ TRIP PLAN READY!")
            print("="*60)
        
        result = {
            'itinerary': optimized_result['itinerary'],
            'schedule': optimized_result['schedule'],
            'total_cost': optimized_result['total_cost'],
            'total_time_hours': optimized_result['total_time'],
            'num_places': optimized_result['num_places'],
            'fitness_score': optimized_result['fitness_score'],
            'budget': budget,
            'budget_remaining': budget - optimized_result['total_cost'],
            'num_days': num_days,
            'adventure_types': adventure_types,
            'travel_month': travel_month,
            'recommendations': top_recommendations
        }
        
        return result
    
    def display_trip_plan(self, result: Dict):
        """
        Display the trip plan in a formatted way
        """
        print("\n📋 TRIP SUMMARY")
        print("-" * 60)
        print(f"Total Places to Visit: {result['num_places']}")
        print(f"Total Cost: ${result['total_cost']:.2f} / ${result['budget']:.2f}")
        print(f"Budget Remaining: ${result['budget_remaining']:.2f}")
        print(f"Total Activity Time: {result['total_time_hours']:.1f} hours")
        print(f"Optimization Score: {result['fitness_score']:.3f}")
        print("-" * 60)
        
        print("\n📅 DAY-BY-DAY ITINERARY")
        print("=" * 60)
        
        for day, places in result['schedule'].items():
            print(f"\n🌅 DAY {day}")
            print("-" * 60)
            
            if not places:
                print("  Rest day / Travel day")
                continue
            
            day_cost = 0
            day_time = 0
            
            for place_name in places:
                # Find place in itinerary
                place_info = None
                for p in result['itinerary']:
                    if p['name'] == place_name:
                        place_info = p
                        break
                
                if place_info:
                    print(f"\n  📍 {place_info['name']}")
                    print(f"     District: {place_info['district']}")
                    print(f"     Category: {place_info['category']}")
                    print(f"     Adventure: {place_info['adventure_type']}")
                    print(f"     Fee: ${place_info['fee']:.2f}")
                    print(f"     Time: {place_info['time']:.1f} hours")
                    
                    day_cost += place_info['fee']
                    day_time += place_info['time']
            
            print(f"\n  💵 Day {day} Cost: ${day_cost:.2f}")
            print(f"  ⏰ Day {day} Time: {day_time:.1f} hours")
        
        print("\n" + "=" * 60)
        print("🎉 Happy Travels in Sri Lanka! 🇱🇰")
        print("=" * 60)
    
    def get_recommendations(self,
                          adventure_types: List[str],
                          budget: float,
                          top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Get top place recommendations without full trip planning
        
        Returns:
            List of (place_name, score) tuples
        """
        # Create user profile
        user_profile = self.preference_learner.create_user_profile(
            adventure_types, budget, 7, self.data_loader.adventure_types_list
        )
        
        # Filter places
        places_by_type = self.data_loader.get_places_by_adventure_type(adventure_types)
        places_by_budget = self.data_loader.get_places_by_budget(budget)
        
        # Find intersection using indices
        available_places = places_by_type[places_by_type['POI Name'].isin(places_by_budget['POI Name'])]
        
        place_names = available_places['POI Name'].tolist()
        
        # Get scores
        scores = []
        for place_name in place_names:
            place_features = self.data_loader.get_feature_vector(place_name)
            score = self.preference_learner.predict_preference(user_profile, place_features)
            scores.append((place_name, score))
        
        # Sort and return top k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# Convenience function
def create_trip_plan(adventure_types: List[str],
                    budget: float,
                    num_days: int,
                    travel_month: int = None,
                    optimization_level: str = 'balanced',
                    data_directory: str = ".",
                    display: bool = True) -> Dict:
    """
    Convenience function to create a trip plan
    
    Args:
        adventure_types: List of preferred adventure types
        budget: Total budget in USD
        num_days: Number of days
        travel_month: Month of travel (1-12)
        optimization_level: 'fast', 'balanced', or 'thorough'
        data_directory: Directory with CSV files
        display: Whether to display the plan
    
    Returns:
        Trip plan dictionary
    """
    planner = SriLankaTripPlanner(data_directory)
    result = planner.plan_trip(adventure_types, budget, num_days, travel_month, optimization_level)
    
    if display:
        planner.display_trip_plan(result)
    
    return result
