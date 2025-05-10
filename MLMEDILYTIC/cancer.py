from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Initialize Flask app
app = Flask(__name__)

# Load the dataset
data = pd.read_csv( "C:/Users/1/Desktop/MEDILYTIC/MLMEDILYTIC/CANCER.csv")
X = data['name']
y = data['suitable_for_cancer']

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Transform the 'name' column into TF-IDF features
X_tfidf = tfidf_vectorizer.fit_transform(X)

# Split data into training and testing sets (if needed later for evaluation)
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Initialize individual models
log_clf = LogisticRegression(random_state=42)
rf_clf = RandomForestClassifier(random_state=42)
svm_clf = SVC(probability=True, random_state=42)

# Ensemble model using VotingClassifier
ensemble_model = VotingClassifier(
    estimators=[
        ('lr', log_clf),
        ('rf', rf_clf),
        ('svc', svm_clf)
    ],
    voting='soft'
)

# Train the model
ensemble_model.fit(X_train, y_train)

# Home page route
@app.route('/')
def index():
    return render_template('index4.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    logging.debug(f"Received form data: {request.form}")
    item_name = request.form.get('fruit_name') or request.form.get('veg_name')
    logging.debug(f"Extracted item name: {item_name}")
    
    if not item_name:
        logging.warning("No item name provided")
        return jsonify({'error': 'No item name provided'}), 400
    
    try:
        # Transform input with TF-IDF Vectorizer
        item_vectorized = tfidf_vectorizer.transform([item_name])
        
        # Make prediction using the ensemble model
        prediction = ensemble_model.predict(item_vectorized)[0]
        
        result = 'Suitable for Cancer Patients' if prediction == 1 else 'Not Suitable for Cancer Patients'
        logging.info(f"Prediction for {item_name}: {result}")
        
        return jsonify({'prediction': result})
    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}", exc_info=True)
        return jsonify({'error': 'An error occurred during prediction'}), 500

# Autocomplete recommendation route
@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    search_term = request.args.get('term')
    
    if search_term:
        # Fetch similar names from the dataset
        recommendations = data[data['name'].str.contains(search_term, case=False, na=False)]['name'].tolist()
    else:
        recommendations = []
    
    # Return matching items as JSON
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True, port=5008)
