# %%
import pandas as pd
from pandas import Series, DataFrame

import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


# %%
"""
## Split Data by average and year (After 1960, Votes number is bigger than 50.000 and avg_vote bigger than 7.0), This is our filter
"""

# %%
split = pd.read_csv("dataset/IMDb movies.csv", low_memory=False)
df = pd.DataFrame(split)
i = df[(df.avg_vote < 7) | (df.year < "1960") | (df.votes < 50000)].index

df.drop(i).to_csv('data.csv', index=False)

# %%
movies = pd.read_csv("data.csv",low_memory=False)

# %%
"""
### Top 250 Movies
"""

# %%
sort_version=pd.read_csv("data.csv",low_memory=False )
sort_version.sort_values(by=['avg_vote'],inplace=True, ascending=False)
top250_movies = sort_version.head(250)

# %%
s = top250_movies['genre'].str.split(', ').apply(Series, 1).stack()
s.index = s.index.droplevel(-1)
s.name = 'genre'
del top250_movies['genre']
split_genres250 = top250_movies.join(s)
split_genres250

# %%
"""
### What are the genre numbers in Top-250 movies?
"""

# %%
genre_counts = (pd.DataFrame(split_genres250.groupby('genre').original_title.nunique())).sort_values('genre', ascending=True)
genre_counts

# %%
genre_counts['original_title'].plot.pie(title= 'Percentage of Movies Genre ' , figsize=(10,10), autopct='%1.1f%%',fontsize=10);

# %%
genre_counts['original_title'].plot.barh(title = 'Movies per Genre',color='LightGreen', figsize=(10, 9));

# %%
"""
# Among the best 250 films determined by votes, there are films in the most drama genre. As it can be understood from here, Drama can be considered as the most liked movie genre.
## Action and Crime genres follow Drama genre.
"""

# %%
"""
#### Split genres in all movies
"""

# %%
ex = movies['genre'].str.split(', ').apply(Series, 1).stack()
ex.index = ex.index.droplevel(-1)
ex.name = 'genre'
del movies['genre']
split_genres = movies.join(ex)

# %%
"""
### What is the average vote density over the years? (All movies with our filters)
"""

# %%
md_year_genre_vote_count = pd.DataFrame(split_genres.groupby(['year','genre'])['avg_vote'].mean())
md_heat_vote_count_pivot = pd.pivot_table(md_year_genre_vote_count, values='avg_vote', index=['genre'], columns=['year'])
sns.set(rc={'figure.figsize':(15,10)})
sns.heatmap(md_heat_vote_count_pivot, linewidths=1, cmap='YlGnBu');

# %%
"""
# As can be seen from the intensity of color, old movies are more appreciated than today's movies.
"""

# %%
"""
#### Get splitted dataset. The roughness of the data has been cleared.
"""

# %%
rec_movies = pd.read_csv("data.csv", low_memory=False)
# Break up the big genre string into a string array
rec_movies['genre'] = rec_movies['genre'].str.split(',')
# Convert genres to string value
rec_movies['genre'] = rec_movies['genre'].fillna("").astype('str')

# %%
"""
### Simple, Content Based (with movie genres) Recommendation System
"""

# %%
tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')

tfidf_matrix = tf.fit_transform(rec_movies['genre'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

titles = rec_movies['original_title']
indices = pd.Series(rec_movies.index, index=rec_movies['original_title'])

recommend_movies = []

# %%
def genre_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    recommend_movies.append(titles[movie_indices])
    

# %%
liste = []
liked_movies = []
users = pd.read_csv("users.csv", low_memory=False)
df = pd.DataFrame(users)
for i in range(3):
    liste.append(df._get_value(i, 1, takeable=True))
    liked_movies = liste[0].split(',')
    print("************ User", i+1 ,"'s liked movies ************")
    for j in range(len(liked_movies)):
        print(liked_movies[j])
        send = liked_movies[j]
        genre_recommendations(send)
    print("\nMovies that the User" ,i+1, "may like.")
    for j in range(len(recommend_movies)):
        print(recommend_movies[j])
        print("\n")
    liste.clear()
    liked_movies.clear()
    recommend_movies.clear()

# %%
"""
## For the 5 movies that the users liked and selected, 10 similar movies are recommended per movie.
"""

# %%


# %%
