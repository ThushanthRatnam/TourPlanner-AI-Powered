"""
Simple test script to verify the trip planner works
"""
import sys
import os
# Add parent directory to path to import src modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.trip_planner import SriLankaTripPlanner

def test_basic_functionality():
    """Test basic functionality of the trip planner"""
    print("Testing Sri Lanka Trip Planner...")
    print("=" * 60)
    
    try:
        # Initialize planner
        print("\n1. Initializing planner...")
        planner = SriLankaTripPlanner(data_directory=".")
        print("   ✅ Planner initialized successfully")
        
        # Test recommendations
        print("\n2. Testing recommendation system...")
        recs = planner.get_recommendations(['Cultural', 'Peace'], budget=300, top_k=5)
        print(f"   ✅ Got {len(recs)} recommendations")
        for i, (place, score) in enumerate(recs[:3], 1):
            print(f"      {i}. {place} (score: {score:.3f})")
        
        # Test trip planning (small scenario for quick testing)
        print("\n3. Testing trip planning...")
        result = planner.plan_trip(
            adventure_types=['Beach'],
            budget=200,
            num_days=3,
            travel_month=3,
            optimization_level='fast',
            verbose=False
        )
        print(f"   ✅ Trip planned successfully")
        print(f"      - Places: {result['num_places']}")
        print(f"      - Cost: ${result['total_cost']:.2f}")
        print(f"      - Fitness: {result['fitness_score']:.3f}")
        
        # Test clustering
        print("\n4. Testing clustering system...")
        cluster_summary = planner.recommender.get_cluster_summary()
        print(f"   ✅ Places grouped into {len(cluster_summary)} clusters")
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nThe system is ready to use. Run demo.py for full demonstrations.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_basic_functionality()
    exit(0 if success else 1)
