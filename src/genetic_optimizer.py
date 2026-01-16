"""
Genetic Algorithm for Trip Optimization
Optimizes the itinerary considering constraints and objectives
"""
import numpy as np
import random
from typing import List, Dict, Tuple, Callable
from copy import deepcopy
import pandas as pd

class TripChromosome:
    """
    Represents a trip itinerary as a chromosome
    Each gene is a place to visit
    """
    
    def __init__(self, places: List[str], num_days: int):
        self.places = places  # List of place names
        self.num_days = num_days
        self.fitness = 0.0
        self.schedule = self._create_schedule()
    
    def _create_schedule(self) -> Dict[int, List[str]]:
        """Organize places into daily schedules"""
        schedule = {day: [] for day in range(1, self.num_days + 1)}
        
        # Distribute places across days
        places_per_day = len(self.places) // self.num_days
        remaining = len(self.places) % self.num_days
        
        idx = 0
        for day in range(1, self.num_days + 1):
            count = places_per_day + (1 if day <= remaining else 0)
            schedule[day] = self.places[idx:idx + count]
            idx += count
        
        return schedule
    
    def get_total_places(self) -> int:
        """Get total number of places in the itinerary"""
        return len(self.places)
    
    def mutate(self, mutation_rate: float = 0.1):
        """Mutate the chromosome by swapping or removing places"""
        if random.random() < mutation_rate:
            if len(self.places) > 1:
                # Swap two random places
                idx1, idx2 = random.sample(range(len(self.places)), 2)
                self.places[idx1], self.places[idx2] = self.places[idx2], self.places[idx1]
                self.schedule = self._create_schedule()


class GeneticAlgorithmOptimizer:
    """
    Genetic Algorithm to optimize trip itineraries
    Considers: budget, time, preferences, weather, diversity
    """
    
    def __init__(self,
                 population_size: int = 100,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.7,
                 elite_size: int = 10,
                 max_generations: int = 100):
        
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.max_generations = max_generations
        self.population = []
        
    def initialize_population(self,
                            available_places: List[str],
                            num_days: int,
                            min_places: int = 3,
                            max_places: int = 15):
        """
        Initialize the population with random chromosomes
        """
        self.population = []
        
        for _ in range(self.population_size):
            # Random number of places for this trip
            num_places = random.randint(
                min(min_places, len(available_places)),
                min(max_places, len(available_places), num_days * 3)
            )
            
            # Random selection of places
            selected_places = random.sample(available_places, num_places)
            
            chromosome = TripChromosome(selected_places, num_days)
            self.population.append(chromosome)
    
    def calculate_fitness(self,
                         chromosome: TripChromosome,
                         data_loader,
                         user_profile: np.ndarray,
                         preference_model,
                         budget: float,
                         current_month: int,
                         weights: Dict[str, float] = None) -> float:
        """
        Calculate fitness score for a chromosome
        
        Fitness components:
        1. User preference score (from neural network)
        2. Budget feasibility
        3. Time feasibility
        4. Weather suitability
        5. Diversity of places
        """
        if weights is None:
            weights = {
                'preference': 0.40,  # Increased - most important
                'budget': 0.20,
                'time': 0.15,
                'weather': 0.10,
                'diversity': 0.15
            }
        
        fitness = 0.0
        total_cost = 0.0
        total_time = 0.0
        weather_scores = []
        categories = set()
        preference_sum = 0.0
        
        # Calculate for each place
        for place_name in chromosome.places:
            place_info = data_loader.get_place_info(place_name)
            
            if not place_info:
                continue
            
            # 1. Preference score from neural network
            place_features = data_loader.get_feature_vector(place_name)
            pref_score = preference_model.predict_preference(user_profile, place_features)
            preference_sum += pref_score
            
            # Track costs and time
            total_cost += place_info['fee']
            total_time += place_info['time']
            
            # Weather score
            weather_score = data_loader.get_weather_score(place_info['district'], current_month)
            weather_scores.append(weather_score)
            
            # Track category diversity
            categories.add(place_info['category'])
        
        num_places = len(chromosome.places)
        if num_places == 0:
            return 0.0
        
        # 1. Average preference score (normalized)
        avg_preference = preference_sum / num_places if num_places > 0 else 0
        fitness += weights['preference'] * avg_preference
        
        # 2. Budget feasibility score
        if total_cost <= budget:
            budget_utilization = total_cost / budget
            # Reward good budget usage (60-95% is ideal)
            if 0.6 <= budget_utilization <= 0.95:
                budget_score = 1.0
            elif budget_utilization < 0.6:
                budget_score = 0.7 + (budget_utilization / 0.6) * 0.3
            else:
                budget_score = 0.9  # Still good if under budget
        else:
            # Penalize over-budget
            budget_score = max(0, 1.0 - (total_cost - budget) / budget)
        fitness += weights['budget'] * budget_score
        
        # 3. Time feasibility score
        available_hours = chromosome.num_days * 8  # Assume 8 hours per day
        if total_time <= available_hours:
            time_utilization = total_time / available_hours
            # Reward good time usage (50-90% is ideal)
            if 0.5 <= time_utilization <= 0.9:
                time_score = 1.0
            elif time_utilization < 0.5:
                time_score = 0.6 + (time_utilization / 0.5) * 0.4
            else:
                time_score = 0.95  # Still good if under limit
        else:
            time_score = max(0, 1.0 - (total_time - available_hours) / available_hours)
        fitness += weights['time'] * time_score
        
        # 4. Weather suitability score
        avg_weather_score = np.mean(weather_scores) if weather_scores else 0.5
        fitness += weights['weather'] * avg_weather_score
        
        # 5. Diversity score
        diversity_score = len(categories) / max(len(chromosome.places), 1)
        fitness += weights['diversity'] * diversity_score
        
        # Bonus: Reward having a good number of places per day (2-4 places/day is ideal)
        places_per_day = num_places / chromosome.num_days
        if 2.0 <= places_per_day <= 4.0:
            density_bonus = 0.3  # 30% bonus for ideal density
        elif 1.5 <= places_per_day <= 5.0:
            density_bonus = 0.15  # 15% bonus for good density
        else:
            density_bonus = -0.1  # Small penalty for poor density
        
        fitness += density_bonus
        
        # Reward more places (don't normalize by dividing by num_places)
        # More places = more value for the trip
        place_count_bonus = min(num_places / (chromosome.num_days * 3), 1.0) * 0.2
        fitness += place_count_bonus
        
        return fitness
    
    def selection(self, fitness_scores: List[float]) -> List[int]:
        """
        Tournament selection - select parents for crossover
        """
        selected_indices = []
        
        for _ in range(self.population_size - self.elite_size):
            # Tournament selection
            tournament_size = 5
            tournament_indices = random.sample(range(len(self.population)), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected_indices.append(winner_idx)
        
        return selected_indices
    
    def crossover(self, parent1: TripChromosome, parent2: TripChromosome) -> Tuple[TripChromosome, TripChromosome]:
        """
        Ordered crossover - create offspring from two parents
        """
        if random.random() > self.crossover_rate:
            return deepcopy(parent1), deepcopy(parent2)
        
        # Get unique places from both parents
        all_places = list(set(parent1.places + parent2.places))
        
        # Create offspring with random subset
        size1 = random.randint(len(parent1.places) // 2, min(len(all_places), len(parent1.places) * 2))
        size2 = random.randint(len(parent2.places) // 2, min(len(all_places), len(parent2.places) * 2))
        
        offspring1_places = random.sample(all_places, min(size1, len(all_places)))
        offspring2_places = random.sample(all_places, min(size2, len(all_places)))
        
        offspring1 = TripChromosome(offspring1_places, parent1.num_days)
        offspring2 = TripChromosome(offspring2_places, parent2.num_days)
        
        return offspring1, offspring2
    
    def evolve(self,
              data_loader,
              user_profile: np.ndarray,
              preference_model,
              budget: float,
              current_month: int,
              verbose: bool = True) -> TripChromosome:
        """
        Run the genetic algorithm to find the best trip itinerary
        
        Returns:
            Best chromosome (trip itinerary)
        """
        best_fitness_history = []
        
        for generation in range(self.max_generations):
            # Calculate fitness for all chromosomes
            fitness_scores = []
            for chromosome in self.population:
                fitness = self.calculate_fitness(
                    chromosome, data_loader, user_profile,
                    preference_model, budget, current_month
                )
                chromosome.fitness = fitness
                fitness_scores.append(fitness)
            
            # Sort population by fitness
            sorted_indices = np.argsort(fitness_scores)[::-1]
            self.population = [self.population[i] for i in sorted_indices]
            fitness_scores = [fitness_scores[i] for i in sorted_indices]
            
            best_fitness = fitness_scores[0]
            best_fitness_history.append(best_fitness)
            
            if verbose and (generation % 10 == 0 or generation == self.max_generations - 1):
                print(f"Generation {generation}: Best Fitness = {best_fitness:.4f}, "
                      f"Avg Fitness = {np.mean(fitness_scores):.4f}")
            
            # Check for convergence
            if generation > 20 and len(set(best_fitness_history[-10:])) == 1:
                if verbose:
                    print("Converged early!")
                break
            
            # Create next generation
            new_population = []
            
            # Elitism - keep best chromosomes
            new_population.extend(self.population[:self.elite_size])
            
            # Selection and crossover
            selected_indices = self.selection(fitness_scores)
            
            for i in range(0, len(selected_indices), 2):
                if i + 1 < len(selected_indices):
                    parent1 = self.population[selected_indices[i]]
                    parent2 = self.population[selected_indices[i + 1]]
                    
                    offspring1, offspring2 = self.crossover(parent1, parent2)
                    
                    # Mutation
                    offspring1.mutate(self.mutation_rate)
                    offspring2.mutate(self.mutation_rate)
                    
                    new_population.extend([offspring1, offspring2])
            
            # Update population
            self.population = new_population[:self.population_size]
        
        # Return best chromosome
        return self.population[0]
    
    def get_optimized_itinerary(self,
                               available_places: List[str],
                               num_days: int,
                               data_loader,
                               user_profile: np.ndarray,
                               preference_model,
                               budget: float,
                               current_month: int,
                               verbose: bool = True) -> Dict:
        """
        Get optimized trip itinerary
        
        Returns:
            Dictionary with itinerary details
        """
        # Calculate reasonable place limits based on trip duration
        # Aim for 2-3 places per day on average
        min_places = max(3, num_days * 2)
        max_places = min(num_days * 4, len(available_places))
        
        if verbose:
            print(f"   Target: {min_places}-{max_places} places for {num_days}-day trip")
        
        # Initialize population
        self.initialize_population(available_places, num_days, min_places, max_places)
        
        # Evolve
        best_chromosome = self.evolve(
            data_loader, user_profile, preference_model,
            budget, current_month, verbose
        )
        
        # Compile results
        total_cost = 0.0
        total_time = 0.0
        itinerary = []
        
        for place_name in best_chromosome.places:
            place_info = data_loader.get_place_info(place_name)
            if place_info:
                total_cost += place_info['fee']
                total_time += place_info['time']
                itinerary.append(place_info)
        
        return {
            'itinerary': itinerary,
            'schedule': best_chromosome.schedule,
            'total_cost': total_cost,
            'total_time': total_time,
            'fitness_score': best_chromosome.fitness,
            'num_places': len(best_chromosome.places)
        }
