"""
Flask API Server for Sri Lanka Trip Planner
Provides REST API endpoints for the frontend
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
from src.trip_planner import SriLankaTripPlanner
import traceback

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Initialize trip planner
planner = None

def initialize_planner():
    """Initialize the trip planner once"""
    global planner
    if planner is None:
        print("Initializing Trip Planner...")
        planner = SriLankaTripPlanner()
        print("Trip Planner initialized successfully!")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Sri Lanka Trip Planner API is running'
    }), 200

@app.route('/api/plan-trip', methods=['POST'])
def plan_trip():
    """Main endpoint to generate trip plan"""
    try:
        # Initialize planner if not already done
        initialize_planner()
        
        # Get request data
        data = request.get_json()
        
        # Validate required fields
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        budget = data.get('budget')
        duration = data.get('duration')
        interests = data.get('interests', [])
        start_location = data.get('startLocation', 'Colombo')
        travel_month = data.get('travelMonth')
        optimization_level = data.get('optimizationLevel', 'balanced')
        
        # Validate inputs
        if not budget or not duration:
            return jsonify({'error': 'Budget and duration are required'}), 400
        
        try:
            budget = float(budget)
            duration = int(duration)
        except ValueError:
            return jsonify({'error': 'Invalid budget or duration format'}), 400
        
        if budget < 100 or budget > 2000:
            return jsonify({'error': 'Budget must be between $100 and $2,000'}), 400
        
        if duration < 3 or duration > 14:
            return jsonify({'error': 'Duration must be between 3 and 14 days'}), 400
        
        # Map frontend interests to planner categories
        interest_mapping = {
            'Beach': 'beach',
            'Cultural': 'cultural',
            'Wildlife': 'wildlife',
            'Hiking': 'hiking',
            'Peace': 'peace',
            'Party': 'party',
            'Urban': 'urban',
            'Adventure': 'adventure'
        }
        
        mapped_interests = [interest_mapping.get(i, i.lower()) for i in interests]
        
        # Handle travel month
        if travel_month:
            try:
                travel_month = int(travel_month)
                if travel_month < 1 or travel_month > 12:
                    travel_month = None
            except (ValueError, TypeError):
                travel_month = None
        else:
            travel_month = None
        
        # Prepare parameters
        adventure_types_list = mapped_interests if mapped_interests else ['cultural', 'beach']
        
        print(f"Processing trip plan request:")
        print(f"  Adventure types: {adventure_types_list}")
        print(f"  Budget: ${budget}")
        print(f"  Duration: {duration} days")
        print(f"  Travel month: {travel_month}")
        print(f"  Optimization: {optimization_level}")
        
        # Generate trip plan
        result = planner.plan_trip(
            adventure_types=adventure_types_list,
            budget=budget,
            num_days=duration,
            travel_month=travel_month,
            optimization_level=optimization_level,
            verbose=True
        )
        
        if result is None:
            return jsonify({
                'error': 'Failed to generate trip plan',
                'message': 'Please try with different preferences'
            }), 500
        
        # Get recommendations with place details
        recommendations_list = []
        for place_name, score in result.get('recommendations', []):
            place_info = planner.data_loader.get_place_info(place_name)
            if place_info:
                recommendations_list.append({
                    'name': place_name,
                    'score': float(score),
                    'district': place_info.get('district', 'N/A'),
                    'category': place_info.get('category', 'N/A'),
                    'adventure_type': place_info.get('adventure_type', 'N/A'),
                    'fee': float(place_info.get('fee', 0)),
                    'time': float(place_info.get('time', 0))
                })
        
        # Format response
        response = {
            'success': True,
            'message': f'Generated {duration}-day trip plan within ${budget} budget',
            'itinerary': result.get('itinerary', []),
            'schedule': result.get('schedule', {}),
            'recommendations': recommendations_list,
            'total_cost': result.get('total_cost', 0),
            'budget_remaining': result.get('budget_remaining', 0),
            'num_places': result.get('num_places', 0),
            'total_time_hours': result.get('total_time_hours', 0),
            'user_preferences': {
                'budget': budget,
                'duration': duration,
                'adventure_types': adventure_types_list,
                'travel_month': travel_month,
                'optimization_level': optimization_level
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"Error generating trip plan: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/api/destinations', methods=['GET'])
def get_destinations():
    """Get list of available destinations"""
    try:
        initialize_planner()
        
        # Get destinations from planner
        destinations = []
        if hasattr(planner, 'df_places') and planner.df_places is not None:
            destinations = planner.df_places[['Place', 'District', 'Category']].to_dict('records')
        
        return jsonify({
            'success': True,
            'destinations': destinations
        }), 200
        
    except Exception as e:
        print(f"Error fetching destinations: {str(e)}")
        return jsonify({
            'error': 'Failed to fetch destinations',
            'message': str(e)
        }), 500

@app.route('/api/interests', methods=['GET'])
def get_interests():
    """Get list of available interest categories"""
    return jsonify({
        'success': True,
        'interests': [
            'Beach',
            'Wildlife',
            'Cultural',
            'Adventure',
            'Historical',
            'Nature',
            'Religious',
            'Photography'
        ]
    }), 200

if __name__ == '__main__':
    print("Starting Sri Lanka Trip Planner API Server...")
    print("Frontend can access the API at: http://localhost:8080")
    app.run(host='0.0.0.0', port=8080, debug=True)
