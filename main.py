import pandas as pd
import numpy as np
import nltk
import math
import matplotlib.pyplot as plt
import seaborn

movies = pd.read_csv('data/AllMoviesDetailsCleaned.csv', sep=';', low_memory=False)

#make genre columns
genres = set()
movies['genres'] = movies['genres'].astype(str)
for row in movies['genres']:
    row = row.split('|')
    for genre in row:
        genres.add(genre)
        
for g in genres:
    movies["is_genre_" + g.replace(" ","_")] = list(map(lambda x : int(g in str(x)), movies["genres"]))
