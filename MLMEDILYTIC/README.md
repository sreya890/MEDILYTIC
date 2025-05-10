  # MEDILYTIC - Healthcare Analytics Platform

MEDILYTIC is a comprehensive healthcare analytics platform that provides various health-related predictions and recommendations using machine learning models. The platform includes multiple modules for different health conditions and nutritional analysis.

## Features

### 1. Diabetes Food Prediction
- Predicts whether a food item is suitable for diabetic patients
- Uses XGBoost classifier for accurate predictions
- Considers Glycemic Index and food characteristics
- Provides instant feedback on food suitability

### 2. Nutrition Recommendation System
- Recommends similar food items based on nutritional profiles
- Uses TF-IDF vectorization and cosine similarity
- Considers multiple nutritional factors:
  - Calcium
  - Protein
  - Carbohydrates
  - Fiber
  - Sugars
  - Fat

### 3. Additional Health Modules
- Cancer Prediction
- PCOD Analysis
- Cardiovascular Disease (CVD) Prediction
- Blood Pressure Analysis

## Technical Stack

- **Backend**: Python, Flask
- **Machine Learning**: 
  - XGBoost
  - Scikit-learn
  - TF-IDF Vectorization
- **Data Processing**: Pandas
- **Frontend**: HTML, CSS, JavaScript

## Project Structure

```
MLMEDILYTIC/
├── app.py                 # Main application file
├── combined_app.py        # Combined features application
├── cancer.py             # Cancer prediction module
├── pcod.py               # PCOD analysis module
├── cvd_f.py              # CVD prediction module
├── bp.py                 # Blood pressure analysis module
├── nut.py                # Nutrition analysis module
├── Models/
│   ├── xgb_classifier.joblib
│   └── tfidf_vectorizer.joblib
└── Data/
    ├── CANCER.csv
    ├── cvd.csv
    ├── HBP.csv
    ├── pcod.csv
    ├── nutrition_1.csv
    └── pred_food.csv
```

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/MEDILYTIC.git
cd MEDILYTIC/MLMEDILYTIC
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python combined_app.py
```

The application will be available at `http://localhost:5000`

## Usage

1. **Diabetes Food Prediction**
   - Enter a food item name
   - Get instant prediction about its suitability for diabetic patients
   - View the Glycemic Index of the food

2. **Nutrition Recommendations**
   - Enter a food item
   - Get recommendations for similar food items based on nutritional profile
   - View detailed nutritional information

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset providers
- Open-source community
- Contributors and maintainers

## Contact

For any queries or support, please open an issue in the repository. 
