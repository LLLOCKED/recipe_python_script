import pandas as pd
from flask import Flask
from flask import jsonify, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

@app.route('/')
def home():
    dataset_path = "preprocessed_recipes.csv"
    df = pd.read_csv(dataset_path)

    user_input_ingredients = request.args.getlist('ingredients')

    # user_input_ingredients = ['eggs', 'milk', 'meat', 'salt'];
    recommended_recipes = recommend_recipes(df, user_input_ingredients, num_recommendations=5)

    return jsonify(pd.DataFrame(recommended_recipes).to_json(orient='records')[1:-1].replace('},{', '} {'))

def recommend_recipes(data, user_input_ingredients, num_recommendations=5):

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(data['ingredients'])

    user_input_vector = tfidf_vectorizer.transform([' '.join(user_input_ingredients)])
    similarities = cosine_similarity(user_input_vector, tfidf_matrix)

    # Визначення найбільш схожих рецептів
    top_recipe_indices = similarities.argsort()[0][::-1][:num_recommendations]
    top_recipes = data.loc[top_recipe_indices, ['name', 'ingredients', 'steps']]

    return top_recipes

# main driver function
if __name__ == '__main__':
	app.run(host='0.0.0.0', port=8080)
