"""
Data Loader and Preprocessing Module for Sri Lanka Trip Planner
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import os

class TripDataLoader:
    """Loads and preprocesses all datasets for the trip planner"""
    
    def __init__(self, data_dir: str = "."):
        self.data_dir = data_dir
        self.places_df = None
        self.distances_df = None
        self.districts_df = None
        self.weather_df = None
        self.load_all_data()
        
    def load_all_data(self):
        """Load all CSV files"""
        # Load adventure category data
        self.places_df = pd.read_csv(
            os.path.join(self.data_dir, "data", "adventure category  - Sri Lanka Trip Planner Data Structure.csv")
        )
        
        # Load distance and time data
        self.distances_df = pd.read_csv(
            os.path.join(self.data_dir, "data", "distance and travelling time - Sri Lanka Trip Planner Data Structure.csv")
        )
        
        # Load districts and attractions
        self.districts_df = pd.read_csv(
            os.path.join(self.data_dir, "data", "Sri Lanka visitable places by Districts  - Sri Lanka Districts and Attractions.csv")
        )
        
        # Load weather data
        self.weather_df = pd.read_csv(
            os.path.join(self.data_dir, "data", "weather condition by month - Sri Lanka Trip Planner Data Structure.csv")
        )
        
        self._preprocess_data()
        
    def _preprocess_data(self):
        """Clean and preprocess the loaded data"""
        # Clean fee column
        self.places_df['Fee (USD)'] = self.places_df['Fee (USD)'].replace(r'[\$,\*]', '', regex=True)
        self.places_df['Fee (USD)'] = pd.to_numeric(self.places_df['Fee (USD)'], errors='coerce').fillna(0)
        
        # Process adventure types
        self.places_df['adventure_types'] = self.places_df['adventure_type'].str.split(',').apply(
            lambda x: [s.strip() for s in x] if isinstance(x, list) else []
        )
        
        # Create adventure type encoding
        all_adventure_types = set()
        for types in self.places_df['adventure_types']:
            all_adventure_types.update(types)
        self.adventure_types_list = sorted(list(all_adventure_types))
        
        # One-hot encode adventure types
        for adv_type in self.adventure_types_list:
            self.places_df[f'adv_{adv_type}'] = self.places_df['adventure_types'].apply(
                lambda x: 1 if adv_type in x else 0
            )
        
        # Add coordinates (estimated for demonstration - in production, use actual coordinates)
        self._add_estimated_coordinates()
        
    def _add_estimated_coordinates(self):
        """Add estimated coordinates for places (simplified)"""
        # Simplified coordinate mapping for districts
        district_coords = {
            'Colombo': (6.9271, 79.8612),
            'Kandy': (7.2906, 80.6337),
            'Galle': (6.0535, 80.2210),
            'Matale': (7.4675, 80.6234),
            'Nuwara Eliya': (6.9497, 80.7891),
            'Hambantota': (6.1429, 81.1212),
            'Matara': (5.9549, 80.5550),
            'Jaffna': (9.6615, 80.0255),
            'Ampara': (7.2974, 81.6728),
            'Kegalle': (7.2513, 80.3464),
            'Badulla': (6.9934, 81.0550),
            'Anuradhapura': (8.3114, 80.4037),
            'Trincomalee': (8.5874, 81.2152)
        }
        
        self.places_df['latitude'] = self.places_df['District'].map(
            lambda x: district_coords.get(x, (7.8731, 80.7718))[0]
        )
        self.places_df['longitude'] = self.places_df['District'].map(
            lambda x: district_coords.get(x, (7.8731, 80.7718))[1]
        )
        
    def get_places_by_adventure_type(self, adventure_types: List[str]) -> pd.DataFrame:
        """Filter places by adventure type"""
        if not adventure_types:
            return self.places_df
        
        mask = pd.Series([False] * len(self.places_df))
        for adv_type in adventure_types:
            if f'adv_{adv_type}' in self.places_df.columns:
                mask = mask | (self.places_df[f'adv_{adv_type}'] == 1)
        
        return self.places_df[mask]
    
    def get_places_by_budget(self, budget: float) -> pd.DataFrame:
        """Filter places within budget"""
        return self.places_df[self.places_df['Fee (USD)'] <= budget]
    
    def get_travel_time(self, origin: str, destination: str) -> float:
        """Get travel time between two locations"""
        # Check direct route
        route = self.distances_df[
            (self.distances_df['Origin'] == origin) & 
            (self.distances_df['Destination'] == destination)
        ]
        
        if not route.empty:
            return route.iloc[0]['Est. Drive Time (Hrs)']
        
        # Check reverse route
        route = self.distances_df[
            (self.distances_df['Origin'] == destination) & 
            (self.distances_df['Destination'] == origin)
        ]
        
        if not route.empty:
            return route.iloc[0]['Est. Drive Time (Hrs)']
        
        # Estimate based on distance if not in database
        return 2.0  # Default estimate
    
    def get_weather_score(self, district: str, month: int) -> float:
        """Get weather suitability score for a district in a given month (1-12)"""
        # Map month to season
        if 1 <= month <= 3:
            season = 'Jan-Mar'
        elif 4 <= month <= 6:
            season = 'Apr-Jun'
        elif 7 <= month <= 9:
            season = 'Jul-Sep'
        else:
            season = 'Oct-Dec'
        
        # Map weather conditions to scores
        weather_scores = {
            'good': 1.0,
            'average': 0.7,
            'medium': 0.5,
            'not good': 0.3
        }
        
        # Find district cluster
        for _, row in self.weather_df.iterrows():
            if district in row['District Cluster']:
                weather_condition = row[season]
                return weather_scores.get(weather_condition, 0.5)
        
        return 0.7  # Default score
    
    def get_feature_vector(self, place_name: str) -> np.ndarray:
        """Get feature vector for a place"""
        place = self.places_df[self.places_df['POI Name'] == place_name]
        
        if place.empty:
            return np.zeros(len(self.adventure_types_list) + 3)
        
        place = place.iloc[0]
        
        # Create feature vector: [adventure_types, fee_normalized, time_normalized, category_encoded]
        features = []
        
        # Adventure type one-hot encoding
        for adv_type in self.adventure_types_list:
            features.append(place[f'adv_{adv_type}'])
        
        # Normalized fee (0-1 scale, assuming max $100)
        features.append(min(place['Fee (USD)'] / 100.0, 1.0))
        
        # Normalized time (0-1 scale, assuming max 8 hours)
        features.append(min(place['Time (Hrs)'] / 8.0, 1.0))
        
        # Category encoding (simplified)
        category_map = {'Cultural': 0.25, 'Nature': 0.5, 'Beach': 0.75, 'Wildlife': 1.0, 'Urban': 0.0, 'Hiking': 0.6}
        features.append(category_map.get(place['Category'], 0.5))
        
        return np.array(features)
    
    def get_all_places(self) -> pd.DataFrame:
        """Return all places"""
        return self.places_df
    
    def get_place_info(self, place_name: str) -> Dict:
        """Get detailed information about a place"""
        place = self.places_df[self.places_df['POI Name'] == place_name]
        
        if place.empty:
            return {}
        
        place = place.iloc[0]
        return {
            'name': place['POI Name'],
            'district': place['District'],
            'category': place['Category'],
            'fee': place['Fee (USD)'],
            'time': place['Time (Hrs)'],
            'adventure_type': place['adventure_type'],
            'latitude': place['latitude'],
            'longitude': place['longitude']
        }
