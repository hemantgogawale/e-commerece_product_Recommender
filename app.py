from flask import Flask, request, render_template
from model import RecommendationModel

app = Flask(__name__)
recommender = RecommendationModel()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_id = request.form.get('username').lower()
    recommendations = recommender.get_recommendations(user_id)
    
    if recommendations is None:
        return render_template('index.html', message="User not found. Please try another username.")
    
    return render_template('index.html', items=recommendations, user=user_id)

if __name__ == '__main__':
    app.run(debug=True)