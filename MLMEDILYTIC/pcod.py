from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

app = Flask(__name__)

data = pd.read_csv('C:/Users/1/Desktop/MEDILYTIC/MLMEDILYTIC/pcod.csv')
X = data['name']
y = data['Suitable for PCOD']

pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', RandomForestClassifier(random_state=42))
])
pipeline.fit(X, y)

@app.route('/')
def index():
    return render_template('index3.html')

@app.route('/predict', methods=['POST'])
def predict():
    item_name = request.form['item_name']
    print(f"Received item name: {item_name}")
    item_vectorized = pipeline.named_steps['vectorizer'].transform([item_name])
    prediction = pipeline.named_steps['classifier'].predict(item_vectorized)[0]
    result = 'Suitable for PCOD Patients' if prediction == 1 else 'Not Suitable for PCOD Patients'
    print(f"Prediction result: {result}")
    return jsonify({'prediction_text': result})

@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    search_term = request.args.get('term')
    recommendations = data[data['name'].str.contains(search_term, case=False, na=False)]['name'].tolist()
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True,port=5050)
