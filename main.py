"""
Interactive Sri Lanka Trip Planner
User-friendly interface for trip planning
"""
from src.trip_planner import SriLankaTripPlanner
from datetime import datetime
import sys

def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def print_section(text):
    """Print a section header"""
    print(f"\n{'─' * 70}")
    print(f"  {text}")
    print(f"{'─' * 70}")

def get_user_input():
    """Get trip preferences from user"""
    print_header(" WELCOME TO SRI LANKA TRIP PLANNER 🇱")
    print("\nPlan your perfect Sri Lankan adventure with AI-powered optimization!")
    print("Using: Neural Networks + Genetic Algorithm + Hybrid Recommendations\n")
    
    # Get number of days
    print_section("📅 TRIP DURATION")
    while True:
        try:
            days_input = input("How many days will you be traveling? (3-14 days): ").strip()
            num_days = int(days_input)
            if 3 <= num_days <= 14:
                break
            else:
                print(" Please enter a number between 3 and 14")
        except ValueError:
            print(" Please enter a valid number")
    
    # Get budget
    print_section("💰 BUDGET")
    print("Budget should cover entrance fees and activities (not including accommodation/food)")
    while True:
        try:
            budget_input = input("What is your total budget in USD? ($100-$2000): ").strip()
            budget = float(budget_input.replace('$', '').replace(',', ''))
            if 100 <= budget <= 2000:
                break
            else:
                print(" Please enter a budget between $100 and $2000")
        except ValueError:
            print(" Please enter a valid number")
    
    # Get adventure types
    print_section("🎯 ADVENTURE PREFERENCES")
    print("\nAvailable adventure types:")
    print("  1. 🏖️  Beach       - Coastal and beach destinations")
    print("  2. 🏛️  Cultural    - Temples, heritage sites, historical places")
    print("  3. 🦁 Wildlife    - National parks, safaris, animal sanctuaries")
    print("  4. 🏔️  Hiking      - Mountains, trekking, scenic trails")
    print("  5. 🙏 Peace       - Relaxation, meditation, tranquil spots")
    print("  6. 🎉 Party       - Nightlife, entertainment, social activities")
    print("  7. 🏙️  Urban       - City experiences, modern attractions")
    print("  8. 🌄 Adventure   - Extreme sports, thrilling experiences")
    
    adventure_map = {
        '1': 'Beach',
        '2': 'Cultural',
        '3': 'Wildlife',
        '4': 'Hiking',
        '5': 'Peace',
        '6': 'Party',
        '7': 'Urban',
        '8': 'Adventure'
    }
    
    print("\nSelect your preferred adventure types (enter numbers separated by commas)")
    print("Example: 1,2,5 for Beach, Cultural, and Peace")
    
    while True:
        selections = input("\nYour choices: ").strip().split(',')
        selections = [s.strip() for s in selections if s.strip()]
        
        if not selections:
            print(" Please select at least one adventure type")
            continue
        
        invalid = [s for s in selections if s not in adventure_map]
        if invalid:
            print(f" Invalid selection(s): {', '.join(invalid)}")
            continue
        
        adventure_types = [adventure_map[s] for s in selections]
        break
    
    # Get travel month
    print_section("📆 TRAVEL MONTH")
    print("\nMonths:")
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for i, month in enumerate(months, 1):
        print(f"  {i:2d}. {month}", end="    ")
        if i % 4 == 0:
            print()
    
    print("\n\nNote: Weather conditions vary by region and season")
    while True:
        try:
            month_input = input("\nSelect travel month (1-12) or press Enter for current month: ").strip()
            if not month_input:
                travel_month = datetime.now().month
                break
            travel_month = int(month_input)
            if 1 <= travel_month <= 12:
                break
            else:
                print(" Please enter a number between 1 and 12")
        except ValueError:
            print(" Please enter a valid number")
    
    # Get optimization level
    print_section("⚙️  OPTIMIZATION SETTINGS")
    print("\n1. 🚀 Fast      - Quick results (30 generations, ~10 seconds)")
    print("2. ⚖️  Balanced - Good quality (80 generations, ~25 seconds) [Recommended]")
    print("3. 🎯 Thorough - Best quality (150 generations, ~60 seconds)")
    
    while True:
        opt_input = input("\nSelect optimization level (1-3) or press Enter for Balanced: ").strip()
        if not opt_input:
            optimization_level = 'balanced'
            break
        
        opt_map = {'1': 'fast', '2': 'balanced', '3': 'thorough'}
        if opt_input in opt_map:
            optimization_level = opt_map[opt_input]
            break
        else:
            print(" Please enter 1, 2, or 3")
    
    return {
        'num_days': num_days,
        'budget': budget,
        'adventure_types': adventure_types,
        'travel_month': travel_month,
        'optimization_level': optimization_level
    }

def display_summary(preferences):
    """Display user preferences summary"""
    print_header("📋 YOUR TRIP PREFERENCES")
    
    months = ['', 'January', 'February', 'March', 'April', 'May', 'June',
              'July', 'August', 'September', 'October', 'November', 'December']
    
    print(f"\n  📅 Duration:     {preferences['num_days']} days")
    print(f"  💰 Budget:       ${preferences['budget']:.2f}")
    print(f"  🎯 Adventures:   {', '.join(preferences['adventure_types'])}")
    print(f"  📆 Travel Month: {months[preferences['travel_month']]}")
    print(f"  ⚙️  Optimization: {preferences['optimization_level'].capitalize()}")
    
    print("\n" + "=" * 70)
    
    confirm = input("\n✓ Proceed with trip planning? (Y/n): ").strip().lower()
    return confirm != 'n'

def main():
    """Main interactive loop"""
    try:
        # Welcome and get user input
        preferences = get_user_input()
        
        # Display summary and confirm
        if not display_summary(preferences):
            print("\n Trip planning cancelled. Run the program again to start over.")
            return
        
        # Initialize planner
        print("\n🔧 Initializing AI models...")
        print("   Loading: Neural Networks, Genetic Algorithm, Recommender System...")
        planner = SriLankaTripPlanner(data_directory=".")
        
        # Plan the trip
        print_header("🧠 AI TRIP PLANNING IN PROGRESS")
        print("\nOur AI is working on your perfect itinerary...")
        print("  ✓ Analyzing your preferences with Neural Networks")
        print("  ✓ Optimizing itinerary with Genetic Algorithm")
        print("  ✓ Ensuring diverse experiences with Hybrid Recommender\n")
        
        result = planner.plan_trip(
            adventure_types=preferences['adventure_types'],
            budget=preferences['budget'],
            num_days=preferences['num_days'],
            travel_month=preferences['travel_month'],
            optimization_level=preferences['optimization_level'],
            verbose=True
        )
        
        # Display the result
        planner.display_trip_plan(result)
        
        # Additional options
        print_section("📊 ADDITIONAL OPTIONS")
        print("\n1. Get more recommendations")
        print("2. Plan another trip")
        print("3. Exit")
        
        choice = input("\nSelect an option (1-3): ").strip()
        
        if choice == '1':
            print_section("🌟 TOP RECOMMENDATIONS")
            recs = planner.get_recommendations(
                adventure_types=preferences['adventure_types'],
                budget=preferences['budget'],
                top_k=10
            )
            
            print("\nTop 10 places matching your preferences:\n")
            for i, (place, score) in enumerate(recs, 1):
                place_info = planner.data_loader.get_place_info(place)
                print(f"{i:2d}. {place} (Score: {score:.3f})")
                print(f"    📍 {place_info['district']} | {place_info['category']}")
                print(f"    💵 ${place_info['fee']} | ⏰ {place_info['time']} hours")
                print(f"    🎯 {place_info['adventure_type']}\n")
        
        elif choice == '2':
            print("\n🔄 Restarting trip planner...\n")
            main()
        
        else:
            print_header("🎉 THANK YOU FOR USING SRI LANKA TRIP PLANNER!")
            print("\nHave an amazing journey! 🌴🇱🇰✨\n")
    
    except KeyboardInterrupt:
        print("\n\n Trip planning cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n An error occurred: {str(e)}")
        print("Please try again or contact support.")
        sys.exit(1)

if __name__ == "__main__":
    main()
