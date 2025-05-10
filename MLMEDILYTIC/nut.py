import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the dataset
df = pd.read_csv("C:/Users/1/Desktop/MEDILYTIC/MLMEDILYTIC/nutrition_1.csv")

# Drop unnecessary columns
df.drop(columns=['Column1', 'serving_size', 'calories'], inplace=True)

# Convert relevant columns to numeric
columns_to_clean = ['calcium', 'protein', 'carbohydrate', 'fiber', 'sugars', 'fat']
for column in columns_to_clean:
    df[column] = df[column].str.replace(' g', '', regex=True).str.replace('mg', '', regex=True).astype(float)

# Convert nutritional profiles to a text format
df['nutritional_profile'] = df[columns_to_clean].astype(str).agg(' '.join, axis=1)

# Initialize TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the data
tfidf_matrix = tfidf_vectorizer.fit_transform(df['nutritional_profile'])

# Calculate cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    given_food_item = request.form['food_item']
    recommended_food_items = recommend_similar_food(given_food_item)

    if isinstance(recommended_food_items, str):
        return render_template('index1.html', error_message=recommended_food_items)  # Display error on same page
    elif recommended_food_items.empty:
        return render_template('index1.html', error_message="No similar food items found.")
    else:
        recommendations = recommended_food_items['name'].tolist()
        return render_template('index1.html', recommendations=recommendations)


@app.route('/suggestions')
def suggestions():
    input_text = request.args.get('input', '')
    if input_text.strip() == '':
        return jsonify([])  # Return an empty list if input is empty or whitespace

    # Get suggestions based on partial input
    suggestions = df[df['name'].str.contains(input_text, case=False)]['name'].tolist()
    return jsonify(suggestions)

def recommend_similar_food(food_item, cosine_sim=cosine_sim, df=df, num_recommendations=2):
    # Check if the input food item is empty
    if not food_item:
        return "Please enter a valid food item."

    # Check if the input food item is present in the database
    if food_item not in df['name'].values:
        return f"Food item '{food_item}' not found."

    # Get the index of the food item
    idx = df[df['name'] == food_item].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_indices = [i[0] for i in sim_scores[1:]]  # Exclude the input food item
    recommended_items = df.iloc[sim_indices[:num_recommendations]]  # Take top num_recommendations items
    return recommended_items

if __name__ == '__main__':
    app.run(debug=True, port=5004)


