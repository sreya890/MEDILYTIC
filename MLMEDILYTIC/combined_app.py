import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

app = Flask(__name__, static_url_path='/Diabetics/static')

# Load datasets
nutrition_df = pd.read_csv("C:/Users/1/Desktop/MEDILYTIC/MLMEDILYTIC/nutrition_1.csv")
diabetes_df = pd.read_csv('C:/Users/1/Desktop/MEDILYTIC/MLMEDILYTIC/pred_food.csv')

# Nutrition recommendation setup
nutrition_df.drop(columns=['Column1', 'serving_size', 'calories'], inplace=True)
columns_to_clean = ['calcium', 'protein', 'carbohydrate', 'fiber', 'sugars', 'fat']
for column in columns_to_clean:
    nutrition_df[column] = nutrition_df[column].str.replace(' g', '', regex=True).str.replace('mg', '', regex=True).astype(float)
nutrition_df['nutritional_profile'] = nutrition_df[columns_to_clean].astype(str).agg(' '.join, axis=1)
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(nutrition_df['nutritional_profile'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Diabetes prediction setup
encoder = OneHotEncoder(sparse_output=False)
X_encoded = encoder.fit_transform(diabetes_df[['Food Name']])
X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(['Food Name']))
X = pd.concat([X_encoded_df, diabetes_df['Glycemic Index']], axis=1)
y = diabetes_df['Suitable for Diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
xgb_classifier = XGBClassifier(random_state=42)
xgb_classifier.fit(X_train, y_train)

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/nutrition')
def nutrition():
    return render_template('index1.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    given_food_item = request.form['food_item']
    recommended_food_items = recommend_similar_food(given_food_item)

    if isinstance(recommended_food_items, str):
        return render_template('index1.html', error_message=recommended_food_items)
    elif recommended_food_items.empty:
        return render_template('index1.html', error_message="No similar food items found.")
    else:
        recommendations = recommended_food_items['name'].tolist()
        return render_template('index1.html', recommendations=recommendations)

@app.route('/suggestions')
def suggestions():
    input_text = request.args.get('input', '')
    if input_text.strip() == '':
        return jsonify([])
    suggestions = nutrition_df[nutrition_df['name'].str.contains(input_text, case=False)]['name'].tolist()
    return jsonify(suggestions)

@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    term = request.args.get('term')
    if term:
        suggestions = diabetes_df[diabetes_df['Food Name'].str.contains(term, case=False)]['Food Name'].tolist()
        return jsonify(suggestions)
    return jsonify([])

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        food_name = request.form['food_name']
        try:
            if food_name not in diabetes_df['Food Name'].values:
                raise ValueError("Invalid food name. Please enter a valid food name from the dataset.")
            food_data = diabetes_df[diabetes_df['Food Name'] == food_name]
            glycemic_index = food_data['Glycemic Index'].values[0]
            input_encoded = encoder.transform([[food_name]])
            input_data = list(input_encoded[0]) + [float(glycemic_index)]
            prediction = xgb_classifier.predict([input_data])[0]
            prediction_text = "Yes, you can have the food" if prediction == 1 else "No, you can't have the food"
            return render_template('index.html', prediction=prediction_text, glycemic_index=glycemic_index)
        except ValueError as e:
            return render_template('index.html', prediction=str(e), glycemic_index=None)
        except Exception as e:
            return render_template('index.html', prediction=f"Error: {str(e)}", glycemic_index=None)

def recommend_similar_food(food_item, cosine_sim=cosine_sim, df=nutrition_df, num_recommendations=2):
    if not food_item:
        return "Please enter a valid food item."
    if food_item not in df['name'].values:
        return f"Food item '{food_item}' not found."
    idx = df[df['name'] == food_item].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_indices = [i[0] for i in sim_scores[1:]]
    recommended_items = df.iloc[sim_indices[:num_recommendations]]
    return recommended_items

if __name__ == '__main__':
    app.run(debug=True, port=5000)
    
