"""
Demo Script - Test the Sri Lanka Trip Planner
Run different scenarios to demonstrate the ML models
"""
import sys
import os
# Add parent directory to path to import src modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.trip_planner import SriLankaTripPlanner, create_trip_plan

def demo_scenario_1():
    """
    Scenario 1: Beach and Party lover with moderate budget
    """
    print("\n" + "🏖️ " * 20)
    print("SCENARIO 1: Beach & Party Adventure")
    print("🏖️ " * 20)
    
    result = create_trip_plan(
        adventure_types=['Beach', 'Party'],
        budget=500,
        num_days=5,
        travel_month=3,  # March - good weather for beaches
        display=True
    )
    
    return result


def demo_scenario_2():
    """
    Scenario 2: Cultural and hiking enthusiast with good budget
    """
    print("\n" + "🏔️ " * 20)
    print("SCENARIO 2: Cultural & Hiking Journey")
    print("🏔️ " * 20)
    
    result = create_trip_plan(
        adventure_types=['Cultural', 'Hiking', 'Peace'],
        budget=800,
        num_days=7,
        travel_month=8,  # August - good for hill country
        display=True
    )
    
    return result


def demo_scenario_3():
    """
    Scenario 3: Wildlife and adventure lover with high budget
    """
    print("\n" + "🦁 " * 20)
    print("SCENARIO 3: Wildlife Safari Adventure")
    print("🦁 " * 20)
    
    result = create_trip_plan(
        adventure_types=['Wildlife', 'Adventure'],
        budget=1200,
        num_days=6,
        travel_month=2,  # February - good for safaris
        display=True
    )
    
    return result


def demo_scenario_4():
    """
    Scenario 4: Budget traveler exploring culture and peace
    """
    print("\n" + "🙏 " * 20)
    print("SCENARIO 4: Budget Cultural Exploration")
    print("🙏 " * 20)
    
    result = create_trip_plan(
        adventure_types=['Cultural', 'Peace'],
        budget=300,
        num_days=4,
        travel_month=7,
        display=True
    )
    
    return result


def demo_recommendations():
    """
    Demo: Get recommendations without full trip planning
    """
    print("\n" + "⭐ " * 20)
    print("QUICK RECOMMENDATIONS")
    print("⭐ " * 20 + "\n")
    
    planner = SriLankaTripPlanner()
    
    # Urban and Party recommendations
    print("🎉 Top places for Urban & Party vibes:")
    print("-" * 50)
    recs = planner.get_recommendations(['Urban', 'Party'], budget=400, top_k=5)
    for i, (place, score) in enumerate(recs, 1):
        place_info = planner.data_loader.get_place_info(place)
        print(f"{i}. {place} (Score: {score:.3f})")
        print(f"   {place_info['district']} - {place_info['category']} - ${place_info['fee']}")
    
    # Nature and Peace recommendations
    print("\n\n🌿 Top places for Nature & Peace:")
    print("-" * 50)
    recs = planner.get_recommendations(['Nature', 'Peace'], budget=600, top_k=5)
    for i, (place, score) in enumerate(recs, 1):
        place_info = planner.data_loader.get_place_info(place)
        print(f"{i}. {place} (Score: {score:.3f})")
        print(f"   {place_info['district']} - {place_info['category']} - ${place_info['fee']}")


def demo_cluster_analysis():
    """
    Demo: Show clustering of places
    """
    print("\n" + "🗂️ " * 20)
    print("PLACE CLUSTERING ANALYSIS")
    print("🗂️ " * 20 + "\n")
    
    planner = SriLankaTripPlanner()
    
    cluster_summary = planner.recommender.get_cluster_summary()
    
    print(f"Places grouped into {len(cluster_summary)} clusters:\n")
    
    for cluster_id, places in cluster_summary.items():
        print(f"\n📦 Cluster {cluster_id} ({len(places)} places):")
        print("-" * 50)
        for place in places[:5]:  # Show first 5
            place_info = planner.data_loader.get_place_info(place)
            print(f"  • {place}")
            print(f"    {place_info['category']} | {place_info['adventure_type']}")
        if len(places) > 5:
            print(f"  ... and {len(places) - 5} more places")


def run_all_demos():
    """
    Run all demonstration scenarios
    """
    print("\n" + "=" * 60)
    print("🌴 SRI LANKA TRIP PLANNER - ML MODEL DEMONSTRATION 🌴")
    print("=" * 60)
    print("\nThis demo showcases:")
    print("  ✅ Neural Networks for preference learning")
    print("  ✅ Genetic Algorithm for trip optimization")
    print("  ✅ Hybrid Recommendation System")
    print("  ✅ K-Means Clustering for place grouping")
    print("=" * 60)
    
    # Run scenarios
    demo_scenario_1()
    input("\n\nPress Enter to continue to next scenario...")
    
    demo_scenario_2()
    input("\n\nPress Enter to continue to next scenario...")
    
    demo_scenario_3()
    input("\n\nPress Enter to continue to next scenario...")
    
    demo_scenario_4()
    input("\n\nPress Enter to see recommendations...")
    
    demo_recommendations()
    input("\n\nPress Enter to see cluster analysis...")
    
    demo_cluster_analysis()
    
    print("\n\n" + "=" * 60)
    print("✅ DEMO COMPLETE!")
    print("=" * 60)
    print("\nThe ML models successfully planned trips using:")
    print("  🧠 Deep Neural Networks for learning preferences")
    print("  🧬 Genetic Algorithms for optimization")
    print("  🎯 Hybrid recommendations with clustering")
    print("\nThese approaches are different from standard algorithms like")
    print("SVM, Random Forest, XGBoost, KNN, Naive Bayes, etc.")
    print("=" * 60)


if __name__ == "__main__":
    # You can run individual scenarios or all demos
    
    # Option 1: Run all demos
    run_all_demos()
    
    # Option 2: Run specific scenario
    # demo_scenario_1()
    
    # Option 3: Just get recommendations
    # demo_recommendations()
