from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv("C:/Users/1/Desktop/MEDILYTIC/MLMEDILYTIC/cvd.csv")

# Features and target variable
X = data['name']
y = data['suitable for cvd']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the model pipeline
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),  # Convert text data to numerical features
    ('classifier', RandomForestClassifier(random_state=42))  # Classifier
])
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{report}')

# Initialize Flask app
app = Flask(__name__)

# Route for the main page
@app.route('/')
def index():
    return render_template('index5.html')

# Route for autocomplete suggestions
@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    term = request.args.get('term', '')
    suggestions = list(data[data['name'].str.contains(term, case=False)]['name'].unique())
    return jsonify(suggestions)

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    fruit_name = request.form.get('fruit_name')
    if fruit_name:
        prediction = pipeline.predict([fruit_name])[0]
        result = 'Suitable for Cardiovascular Patients' if prediction == 1 else 'Not Suitable for Cardiovascular Patients'
        return jsonify({'prediction': result})
    return jsonify({'prediction': 'No prediction available'})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True,port=5011)
