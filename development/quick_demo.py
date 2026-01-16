"""
Quick Demo Script for Sri Lanka Trip Planner
Generates results and graphs quickly for demonstrations
"""
import sys
import os
# Add parent directory to path to import src modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
from src.trip_planner import create_trip_plan
from src.visualization import ModelVisualizer
from src.model_evaluator import ModelEvaluator
from src.xai_explainer import XAIExplainer

def quick_demo():
    """Run a quick demonstration with graphs"""
    
    print("=" * 70)
    print("  🚀 QUICK DEMO - Sri Lanka Trip Planner")
    print("=" * 70)
    
    # 1. Quick Trip Planning Demo
    print("\n1️⃣  Generating Sample Trip Plans...")
    start = time.time()
    
    result = create_trip_plan(
        num_days=5,
        budget=500,
        adventure_types=['Beach', 'Cultural'],
        travel_month=7,
        optimization_level='fast',  # Fast for demo
        verbose=False
    )
    
    print(f"✓ Trip planned in {time.time() - start:.2f}s")
    print(f"  - Places: {result['num_places']}")
    print(f"  - Cost: ${result['total_cost']:.2f}")
    print(f"  - Fitness Score: {result['fitness_score']:.3f}")
    
    # 2. Generate Visualizations
    print("\n2️⃣  Generating Visualizations...")
    viz = TripPlannerVisualizer()
    
    # Quick plots
    print("   📊 Creating performance graphs...")
    viz.plot_training_history(result['planner'])
    print("   ✓ Saved: plots/training_history.png")
    
    viz.plot_genetic_convergence(result['planner'])
    print("   ✓ Saved: plots/genetic_convergence.png")
    
    viz.plot_place_clustering(result['planner'])
    print("   ✓ Saved: plots/place_clustering.png")
    
    # 3. Quick Model Evaluation
    print("\n3️⃣  Running Model Evaluation...")
    evaluator = ModelEvaluator(result['planner'])
    
    # Use small test size for speed
    metrics = evaluator.evaluate_recommendation_system(
        test_size=0.2,
        n_samples=50  # Small sample for speed
    )
    
    print("\n   📈 Recommendation Metrics:")
    print(f"   - Precision@5: {metrics['precision_at_5']:.3f}")
    print(f"   - Recall@5: {metrics['recall_at_5']:.3f}")
    print(f"   - NDCG@5: {metrics['ndcg_at_5']:.3f}")
    
    # Save evaluation plots
    evaluator.save_evaluation_plots('plots/')
    print("   ✓ Saved: plots/evaluation_metrics.png")
    print("   ✓ Saved: plots/confusion_matrix.png")
    
    # 4. Feature Importance (Quick XAI)
    print("\n4️⃣  Explainability Analysis...")
    explainer = XAIExplainer(result['planner'])
    
    importance = explainer.get_feature_importance()
    print("\n   🔍 Top 5 Important Features:")
    for i, (feature, score) in enumerate(importance[:5], 1):
        print(f"   {i}. {feature}: {score:.3f}")
    
    explainer.plot_feature_importance(save_path='plots/feature_importance.png')
    print("   ✓ Saved: plots/feature_importance.png")
    
    print("\n" + "=" * 70)
    print("  ✅ DEMO COMPLETE!")
    print("=" * 70)
    print("\n📁 All graphs saved in: plots/")
    print("\n📊 Generated files:")
    print("   - plots/training_history.png")
    print("   - plots/genetic_convergence.png")
    print("   - plots/place_clustering.png")
    print("   - plots/evaluation_metrics.png")
    print("   - plots/confusion_matrix.png")
    print("   - plots/feature_importance.png")
    print("\n" + "=" * 70)

if __name__ == "__main__":
    quick_demo()
