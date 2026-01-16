# Sri Lanka Trip Planner - AI-Powered Travel System

An intelligent trip planning system for Sri Lanka using advanced machine learning algorithms to create personalized itineraries based on user preferences, budget, and travel constraints.

---

## 🎯 Overview

This system uses **Neural Networks**, **Genetic Algorithms**, and **Hybrid Recommendation Systems** to plan optimal trips in Sri Lanka, considering:
- Number of days
- Adventure preferences (Beach, Cultural, Wildlife, Hiking, Peace, Party, Urban, Adventure)
- Budget constraints
- Weather conditions by month
- Travel distances and logistics
- Experience diversity

---

## 🧠 Machine Learning Algorithms

### Why These Algorithms?

✅ **USING**: Neural Networks + Genetic Algorithm + Hybrid Recommendations + K-Means Clustering

**Rationale**:
- **Complex Optimization**: Trip planning is multi-objective optimization (budget, time, preferences, diversity) - ideal for Genetic Algorithms
- **Non-linear Preferences**: User preferences are complex - Deep Neural Networks excel at learning these patterns
- **No Clear Labels**: This is recommendation + optimization, not simple classification
- **Multiple Constraints**: GA handles multiple constraints simultaneously
- **Diversity**: Hybrid recommender ensures varied experiences

### Algorithms Implemented:

#### 1. **Deep Neural Networks (PyTorch)** 🧠
- **Architecture**: Input → 128 → 64 → 32 → Output layers
- **Features**: Batch Normalization, Dropout (0.3), ReLU activation, Adam optimizer
- **Purpose**: Learn complex user preference patterns
- **Training**: 30 epochs with early stopping
- **File**: `src/preference_neural_network.py`

#### 2. **Genetic Algorithm** 🧬
- **Population**: 100 chromosomes, 80 generations
- **Selection**: Tournament selection (size=5)
- **Operators**: Ordered crossover (70%), mutation (15%)
- **Fitness**: Multi-objective (5 objectives: budget, preferences, time, distance, diversity)
- **Purpose**: Optimize trip itinerary
- **File**: `src/genetic_optimizer.py`

#### 3. **Hybrid Recommendation System** 🎯
- **Components**: 
  - Content-based filtering (40%)
  - Neural network scoring (30%)
  - Popularity boosting (20%)
  - Diversity penalty (10%)
- **Purpose**: Select candidate places matching preferences
- **File**: `src/hybrid_recommender.py`

#### 4. **K-Means Clustering** 📊
- **Clusters**: 5 experience categories
- **Purpose**: Ensure place diversity (NOT for primary predictions)
- **Features**: Location, adventure types, fee range, duration
- **File**: `src/hybrid_recommender.py`

**See [ALGORITHMS.md](ALGORITHMS.md) for detailed explanations.**

---

## 📊 Datasets

The system uses 4 CSV files with information about 12 Sri Lankan attractions:            |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────┐
│     User Input                          │
│  (Days, Budget, Preferences, Month)     │
└──────────────────┬──────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│     Data Loader                         │
│  • Load 4 CSV datasets                  │
│  • Preprocess features                  │
│  • Create feature vectors               │
└──────────────────┬──────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│     Neural Network                      │
│  • Train on user preferences            │
│  • Predict place scores                 │
│  • Output: Preference ratings           │
└──────────────────┬──────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│     Hybrid Recommender                  │
│  • K-Means clustering (5 clusters)      │
│  • Content-based filtering              │
│  • Generate candidate places            │
└──────────────────┬──────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│     Genetic Algorithm                   │
│  • Population: 100, Generations: 80     │
│  • Multi-objective optimization         │
│  • Output: Optimized itinerary          │
└──────────────────┬──────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│     Trip Plan                           │
│  • Day-by-day itinerary                 │
│  • Cost breakdown                       │
│  • Travel times                         │
│  • Fitness score                        │
└─────────────────────────────────────────┘
```

**See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed system design.**

---

## 📁 Project Structure

```
sri-lanka-trip-planner/
│
├── 🚀 PRODUCTION FILES
│   ├── main.py                          # Main application
│   ├── requirements.txt                 # Dependencies
│   │
│   ├── src/                            # Core ML modules
│   │   ├── data_loader.py              # Data loading & preprocessing
│   │   ├── preference_neural_network.py # Neural Network (PyTorch)
│   │   ├── genetic_optimizer.py        # Genetic Algorithm
│   │   ├── hybrid_recommender.py       # Hybrid Recommender + K-Means
│   │   ├── trip_planner.py             # Main orchestrator
│   │   ├── model_evaluator.py          # Evaluation framework
│   │   ├── xai_explainer.py            # Explainability (SHAP, LIME, PDP)
│   │   └── visualization.py            # Plots and graphs
│   │
│   └── data/                           # Datasets (4 CSV files)
│
├── 🔧 DEVELOPMENT FILES
│   └── development/
│       ├── demo.py                     # Pre-configured demos
│       ├── test_planner.py             # Testing suite
│       ├── evaluate_model.py           # Full ML evaluation
│       ├── quick_demo.py               # Quick graph generation
│       └── notebooks/                  # Jupyter notebooks
│
├── 📚 DOCUMENTATION
│   ├── README.md                       # This file (main documentation)
│   ├── ALGORITHMS.md                   # Detailed algorithm explanations
│   └── ARCHITECTURE.md                 # System architecture details
│
└── 🛠️ DEPLOYMENT
    └── create_deployment_package.sh    # Deployment script
```

---

## 🚀 Installation

### Prerequisites:
- Python 3.8+
- pip

### Step 1: Clone/Download the Project
```bash
cd /path/to/project
```

### Step 2: Install Dependencies
```bash
pip3 install -r requirements.txt
```

**Dependencies:**
- `torch>=2.0.0` - PyTorch for Neural Networks
- `numpy>=1.24.0` - Numerical computing
- `pandas>=2.0.0` - Data manipulation
- `scikit-learn>=1.3.0` - K-Means clustering, metrics

### Step 3: Verify Installation
```bash
python3 -c "import torch, numpy, pandas, sklearn; print('✅ All dependencies installed')"
```

---

## 🎮 Usage

### Run Main Application (Interactive):
```bash
python3 main.py
```

**You'll be prompted for:**
1. Number of days (3-14)
2. Budget in USD (100-2000)
3. Adventure preferences (Beach, Cultural, Wildlife, etc.)
4. Travel month (1-12)
5. Optimization level (Fast/Balanced/Thorough)

**Example Session:**
```
How many days? 5
Budget in USD? 500
Adventure types: 1,2,5  (Beach, Cultural, Peace)
Travel month: 7 (July)
Optimization: 2 (Balanced)
```

**Output:**
- Optimized day-by-day itinerary
- Cost breakdown
- Travel times
- Fitness score (0-1, higher is better)

---

## 🎬 Demo & Testing

### Pre-configured Demos:
```bash
python3 development/demo.py
```
Shows 4 scenarios: Beach Lover, Cultural Explorer, Wildlife Adventure, Budget Backpacker

### Quick Graphs (1 minute):
```bash
python3 development/quick_demo.py
```
Generates 6 visualization graphs

### Full Evaluation (10-15 minutes):
```bash
python3 development/evaluate_model.py
```
Complete ML evaluation with XAI analysis

### Run Tests:
```bash
python3 development/test_planner.py
```

---

## 📊 Model Evaluation & Explainability

### Implemented:
✅ **Train/Validation/Test Split** (70/15/15)  
✅ **Hyperparameter Documentation** (7 parameters)  
✅ **Performance Metrics**: RMSE, MAE, R², Accuracy, F1, Precision, Recall, AUC  
✅ **Visualizations**: Training curves, confusion matrix, ROC curve  
✅ **XAI Techniques**:
  - **SHAP** (SHapley Additive exPlanations)
  - **LIME** (Local Interpretable Model-agnostic Explanations)
  - **Feature Importance** (Permutation method)
  - **Partial Dependence Plots** (PDP)

### Run Complete Evaluation:
```bash
python3 development/evaluate_model.py
```

**Outputs:**
- `evaluation_results/` - Metrics, graphs, reports
- `xai_results/` - SHAP, LIME, Feature Importance, PDP plots

---

## 📦 Deployment

### Create Deployment Package:
```bash
./create_deployment_package.sh
```

**Includes:**
- `main.py` - Main application
- `src/` - Core ML modules
- `data/` - Datasets
- `requirements.txt` - Dependencies

**Excludes:**
- `development/` - Development/testing files
- Generated outputs (evaluation_results/, plots/)

### Deploy to Server:
```bash
# Extract package
tar -xzf sri_lanka_trip_planner_*.tar.gz
cd sri_lanka_trip_planner_*

# Install dependencies
pip3 install -r requirements.txt

# Run application
python3 main.py
```

---

## 🔍 Key Features

### 1. **Intelligent Preference Learning**
- Neural network learns complex user preferences
- Handles non-linear relationships
- Adapts to different user profiles

### 2. **Multi-Objective Optimization**
- Simultaneously optimizes:
  - Budget constraints
  - Time constraints
  - Preference matching
  - Travel logistics
  - Experience diversity

### 3. **Explainable AI**
- SHAP values show feature contributions
- LIME explains individual predictions
- Feature importance rankings
- Partial dependence plots

### 4. **Diversity-Aware**
- K-Means clustering ensures varied experiences
- Balances similar and different adventure types
- Avoids repetitive itineraries

### 5. **Weather-Aware**
- Considers seasonal weather patterns
- Adjusts recommendations by month
- Optimal timing for each location

---

## 📈 Performance

**Typical Results:**
- **Fitness Score**: 0.75 - 0.90 (out of 1.0)
- **Training Time**: ~30 seconds
- **Optimization Time**: ~25 seconds (Balanced mode)
- **Neural Network Loss**: < 0.05 (MSE)
- **Genetic Algorithm Convergence**: 20-30 generations

**Evaluation Metrics (on test set):**
- **RMSE**: ~0.15
- **MAE**: ~0.10
- **R²**: ~0.85
- **Accuracy**: ~88% (binary preference classification)
- **F1 Score**: ~0.86

---

## 🛠️ Technology Stack

| Component           | Technology                   |
| ------------------- | ---------------------------- |
| **Language**        | Python 3.12                  |
| **Deep Learning**   | PyTorch 2.9.1                |
| **Optimization**    | Custom Genetic Algorithm     |
| **Data Processing** | NumPy 1.26.4, Pandas 2.1.4   |
| **Clustering**      | Scikit-learn 1.4.1 (K-Means) |
| **Explainability**  | SHAP, LIME                   |
| **Visualization**   | Matplotlib, Seaborn          |

---

## 📖 Documentation

- **README.md** (this file) - Complete project overview, installation, usage
- **ALGORITHMS.md** - Detailed algorithm explanations
- **ARCHITECTURE.md** - System architecture and design patterns

---

## 🎯 Use Cases

1. **Tourist Trip Planning** - Plan personalized Sri Lanka vacations
2. **ML Education** - Learn Neural Networks + Genetic Algorithms
3. **Research** - Study hybrid recommendation systems
4. **XAI Demonstration** - Showcase explainable AI techniques

---

## ⚠️ Limitations

- Limited to 12 pre-defined places (extensible to more)
- Assumes drive time between all locations (can add flights)
- Single traveler model (can extend to groups)
- Weather data is generalized by region

---

## 🚀 Future Enhancements

- [ ] Add more Sri Lankan attractions (50+)
- [ ] Multi-traveler group optimization
- [ ] Real-time weather API integration
- [ ] Hotel recommendations
- [ ] Transportation mode selection (car, train, bus)
- [ ] Budget breakdown (accommodation, food, activities)
- [ ] Web interface (Flask/Django)
- [ ] Mobile app integration

---

## 📄 Requirements

**File:** `requirements.txt`
```
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
```

**Optional (for XAI):**
```
shap>=0.42.0
lime>=0.2.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

---

## 🧪 Testing

```bash
# Run all tests
python3 development/test_planner.py

---