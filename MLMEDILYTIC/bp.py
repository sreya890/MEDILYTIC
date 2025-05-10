from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# Initialize Flask app
app = Flask(__name__)

# Load the dataset and train the model
data = pd.read_csv("C:/Users/1/Desktop/MEDILYTIC/MLMEDILYTIC/HBP.csv")
print("Columns available in the dataset:", data.columns)

# Check for the existence of the 'Food Name' column
if 'Food Name' in data.columns:
    X = data['Food Name']
else:
    raise ValueError("'Food Name' column not found in the dataset")

# Assign the target variable
y = data['Suitable for Blood Pressure']

# Create and train the pipeline
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', RandomForestClassifier(random_state=42))
])
pipeline.fit(X, y)

# Home page route
@app.route('/')
def index():
    return render_template('index2.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()  # Get the JSON data from the request
        item_name = data.get('item_name')  # Extract the item_name
        
        if item_name is None:
            raise ValueError("Missing 'item_name' in request data")

        item_vectorized = pipeline.named_steps['vectorizer'].transform([item_name])
        prediction = pipeline.named_steps['classifier'].predict(item_vectorized)[0]
        
        return jsonify({'prediction': 'Suitable' if prediction == 1 else 'Not Suitable'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400  # Return error message and HTTP status 400

# Modify the autocomplete route
@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    search_term = request.args.get('term', '').lower()
    
    # Fetch similar food names from dataset
    recommendations = data[data['Food Name'].str.lower().str.contains(search_term)]['Food Name'].tolist()
    
    # Limit to top 10 recommendations
    recommendations = recommendations[:10]
    
    # Return matching items as JSON
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True, port=5019)
