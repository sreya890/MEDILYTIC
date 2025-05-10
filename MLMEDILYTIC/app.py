from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

app = Flask(__name__, static_url_path='/Diabetics/static')

# Load the dataset
df = pd.read_csv('C:/Users/1/Desktop/MEDILYTIC/MLMEDILYTIC/pred_food.csv')

# Preprocessing
encoder = OneHotEncoder(sparse_output=False)
X_encoded = encoder.fit_transform(df[['Food Name']])
X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(['Food Name']))
X = pd.concat([X_encoded_df, df['Glycemic Index']], axis=1)
y = df['Suitable for Diabetes']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the XGBoost model
xgb_classifier = XGBClassifier(random_state=42)
xgb_classifier.fit(X_train, y_train)

# Endpoint for autocomplete suggestions
# Assuming Flask is initialized and the DataFrame (df) is already loaded

@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    term = request.args.get('term', '').lower()
    print(f"Received autocomplete request for term: {term}")  # Debug print
    if term:
        suggestions = df[df['Food Name'].str.lower().str.contains(term)]['Food Name'].tolist()
        print(f"Suggestions: {suggestions[:5]}")  # Debug print
        return jsonify(suggestions[:5])  # Limit to 5 suggestions
    return jsonify([])


# Define routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the input data from the form
        food_name = request.form['food_name']

        try:
            # Check if the food name is in the dataset
            if food_name not in df['Food Name'].values:
                raise ValueError("Invalid food name. Please enter a valid food name from the dataset.")

            # Get the glycemic index for the food
            food_data = df[df['Food Name'] == food_name]
            glycemic_index = food_data['Glycemic Index'].values[0]

            # Transform the input data
            input_encoded = encoder.transform([[food_name]])
            input_data = list(input_encoded[0]) + [float(glycemic_index)]

            # Make prediction
            prediction = xgb_classifier.predict([input_data])[0]

            # Convert prediction to text
            prediction_text = "Yes, you can have the food" if prediction == 1 else "No, you can't have the food"
            
            return render_template('index.html', prediction=prediction_text, glycemic_index=glycemic_index)
        except ValueError as e:
            # Handle known exceptions
            return render_template('index.html', prediction=str(e), glycemic_index=None)
        except Exception as e:
            # Handle other exceptions
            return render_template('index.html', prediction=f"Error: {str(e)}", glycemic_index=None)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
