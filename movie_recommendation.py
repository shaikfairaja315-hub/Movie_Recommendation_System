import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv("movies.csv")

# Convert genres into vectors
cv = CountVectorizer()
count_matrix = cv.fit_transform(movies['genre'])

# Calculate similarity
similarity = cosine_similarity(count_matrix)

# Recommendation function
def recommend(movie_name):
    movie_index = movies[movies['title'] == movie_name].index[0]
    scores = list(enumerate(similarity[movie_index]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

    print(f"\nMovies recommended for {movie_name}:\n")

    for movie in sorted_scores[1:4]:
        print(movies.iloc[movie[0]].title)

# Test recommendation
recommend("Inception")