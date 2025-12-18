import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

class RecommendationModel:
    def __init__(self):
        # Load the saved models and data
        self.sentiment_model = pickle.load(open('models/sentiment_classification_xgboost_model.pkl', 'rb'))
        self.tfidf_vectorizer = pickle.load(open('models/tfidf_vectorizer.pkl', 'rb'))
        self.user_final_rating = pickle.load(open('models/user_final_rating.pkl', 'rb'))
        self.clean_data = pickle.load(open('models/cleaned_data.pkl', 'rb')) # Cleaned dataset with reviews

    def get_recommendations(self, user_id):
        if user_id not in self.user_final_rating.index:
            return None
        
        # 1. Get top 20 products from recommendation engine
        top20_products = self.user_final_rating.loc[user_id].sort_values(ascending=False)[0:20].index
        
        # 2. Filter data for these 20 products
        df_top20 = self.clean_data[self.clean_data['id'].isin(top20_products)]
        
        # 3. Predict sentiment for each review of these 20 products
        # Combine text and title as done in training
        df_top20['combined_text'] = df_top20['reviews_text'] + " " + df_top20['reviews_text_cleaned']
        X = self.tfidf_vectorizer.transform(df_top20['combined_text'].values.astype('U'))
        df_top20['predicted_sentiment'] = self.sentiment_model.predict(X)
        
        # 4. Calculate % of positive sentiment for each product
        product_sentiment = df_top20.groupby('id')['predicted_sentiment'].agg(['sum', 'count'])
        product_sentiment['positive_percentage'] = (product_sentiment['sum'] / product_sentiment['count']) * 100
        
        # 5. Join with product names and sort
        top5_ids = product_sentiment.sort_values(by='positive_percentage', ascending=False).head(5).index
        top5_products = self.clean_data[self.clean_data['id'].isin(top5_ids)][['name']].drop_duplicates()
        
        return top5_products['name'].tolist()