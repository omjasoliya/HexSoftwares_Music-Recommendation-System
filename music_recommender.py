import pandas as pd
from flask import Flask, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split, cross_validate

app = Flask(__name__)

# Data Loading and Preprocessing
def load_and_preprocess_data():
    user_data = pd.read_csv('data/user_interactions.csv')
    song_data = pd.read_csv('data/song_metadata.csv')
    audio_data = pd.read_csv('data/audio_features.csv')

    # Merge datasets on song ID
    data = pd.merge(user_data, song_data, on='song_id')
    data = pd.merge(data, audio_data, on='song_id')

    # Fill missing values and scale features
    data.fillna(0, inplace=True)
    scaler = StandardScaler()
    data[['tempo', 'loudness', 'danceability']] = scaler.fit_transform(data[['tempo', 'loudness', 'danceability']])
    
    return data

# Collaborative Filtering Model
def collaborative_filtering(data):
    reader = Reader(rating_scale=(1, 5))
    dataset = Dataset.load_from_df(data[['user_id', 'song_id', 'rating']], reader)
    trainset, testset = train_test_split(dataset, test_size=0.2)

    algo = SVD()
    algo.fit(trainset)
    
    return algo

# Content-Based Filtering
def content_based_filtering(data):
    song_features = data[['tempo', 'loudness', 'danceability', 'genre']]
    song_similarity = cosine_similarity(song_features)
    return song_similarity

def recommend_songs(song_id, song_similarity, data, top_n=10):
    song_index = data[data['song_id'] == song_id].index[0]
    similar_songs = sorted(list(enumerate(song_similarity[song_index])), key=lambda x: x[1], reverse=True)[1:top_n+1]
    recommended_songs = [data['song_id'].iloc[i[0]] for i in similar_songs]
    return recommended_songs

# Hybrid Recommendation
def hybrid_recommendation(user_id, song_id, data):
    collaborative_model = collaborative_filtering(data)
    song_similarity = content_based_filtering(data)

    # Collaborative filtering recommendation
    collaborative_prediction = collaborative_model.predict(user_id, song_id)

    # Content-based filtering recommendation
    content_recommendations = recommend_songs(song_id, song_similarity, data)
    
    return content_recommendations

# Flask API Endpoint
@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = int(request.args.get('user_id'))
    song_id = int(request.args.get('song_id'))
    
    data = load_and_preprocess_data()
    recommendations = hybrid_recommendation(user_id, song_id, data)
    
    return jsonify({'recommendations': recommendations})

if __name__ == '__main__':
    app.run(debug=True)
