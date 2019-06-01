import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 


movies = pd.read_csv('data/AllMoviesDetailsCleaned.csv', sep=';', low_memory=False)
cast = pd.read_csv('data/AllMoviesCastingRaw.csv', sep=';', low_memory=False)
directors = pd.read_csv('data/900_acclaimed_directors_awards.csv', sep=';', low_memory=False)

movies.popularity = pd.to_numeric(movies.popularity.str.replace(',',''))

#create genre variables
movies =  movies[movies.revenue > 0]
genres = set()
movies['genres'] = movies['genres'].astype(str)
for row in movies['genres']:
    row = row.split('|')
    for genre in row:
        genres.add(genre)
        
for g in genres:
    movies["is_genre_" + g.replace(" ","_")] = list(map(lambda x : int(g in str(x)), movies["genres"]))

#create star_power variable
movies = pd.merge(movies, cast, on='id')
movies['star_power'] = 0

for row_num in range(0, movies.shape[0]): 
    if movies.director_name.iloc[row_num] in list(directors['name']):
        movies.star_power.iloc[row_num] = directors['Total awards'].iloc[(np.where(movies.director_name.iloc[row_num] == directors['name'])[0][0])]

actors = {}

for row_num in range(0, movies.shape[0]):
    for column in ['actor1_name', 'actor2_name', 'actor3_name', 'actor4_name', 'actor5_name']:
        if movies[column].iloc[row_num] not in actors:
            actors[movies[column].iloc[row_num]] = 1
        else:
            actors[movies[column].iloc[row_num]] += 1

actors['none'] = 0
for row_num in range(0, movies.shape[0]): 
    movies.star_power.iloc[row_num] = movies.star_power.iloc[row_num] + (actors[movies.actor1_name.iloc[row_num]] + 
                                                                         actors[movies.actor2_name.iloc[row_num]] +
                                                                         actors[movies.actor3_name.iloc[row_num]] +
                                                                         actors[movies.actor4_name.iloc[row_num]] +
                                                                         actors[movies.actor5_name.iloc[row_num]])

#create blockbuster_season and oscar_season variables
movies.release_date = pd.to_datetime(movies.release_date, format='%d/%m/%Y')

movies['blockbuster_season'] = 0
movies['oscar_season'] = 0

for row_num in range(0, movies.shape[0]):
    if movies.release_date.iloc[row_num].month in [6, 7, 8]: # blockbuster season
        movies['blockbuster_season'].iloc[row_num] = 1
    if movies.release_date.iloc[row_num].month in [11, 12]: # oscar season
        movies['oscar_season'].iloc[row_num] = 1


#DO DATA ANALYSIS





#train test split
movies2 = movies.sample(frac=1) # shuffle movies
movies2 = movies2[movies2.columns[[1, 7, 11, 12, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 
                42, 61, 62, 63]]].dropna().reset_index(drop=True)
train_x = movies2.drop(['revenue'], axis=1).iloc[:6000]
train_y = movies2.revenue[:6000]
test_x = movies2.drop(['revenue'], axis=1).iloc[6000:]
test_y = movies2.revenue[6000:]

###regression
#linear regression
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

regr = linear_model.LinearRegression()

regr.fit(train_x, train_y)
predictions = regr.predict(test_x)

print("Mean squared error: %.2f"
      % mean_squared_error(test_y, predictions))

# lasso
lasso = linear_model.Lasso(alpha=.5)

lasso.fit(train_x, train_y)
predictions = lasso.predict(test_x)
print("Mean squared error: %.2f"
      % mean_squared_error(test_y, predictions))



###classification
#create revenue classes
movies2['class'] = 0

for row_num in range(0, movies2.shape[0]):
    if movies2.revenue.iloc[row_num] > 100000000:
        movies2['class'].iloc[row_num] = 3
    elif movies2.revenue.iloc[row_num] > 50000000:
        movies2['class'].iloc[row_num] = 2
    else:
        movies2['class'].iloc[row_num] = 1

#train test split
train_x = movies2.drop(['revenue', 'class'], axis=1).iloc[:6000]
train_y = movies2['class'].iloc[:6000]
test_x = movies2.drop(['revenue', 'class'], axis=1).iloc[6000:]
test_y = movies2['class'].iloc[6000:]

#DO RANDOM FOREST
# Import the model we are using
from sklearn.ensemble import RandomForestRegressor

random_forest = RandomForestRegressor(n_estimators = 1000, random_state = 42)

# Train the model on training data
random_forest.fit(train_x, train_y);

# Use the forest's predict method on the test data
predictions = random_forest.predict(test_x)

# Calculate the absolute errors
errors = abs(predictions - test_y)

# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')



#DO KNN
from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
scaler.fit(train_x)

X_train = scaler.transform(train_x)  
X_test = scaler.transform(test_x) 

from sklearn.neighbors import KNeighborsClassifier  
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, train_y)
y_pred = knn.predict(X_test)

from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(test_y, y_pred))

error = []

# Calculating error for K values between 1 and 40
for i in range(1, 40):  
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,train_y)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != test_y))
    
plt.figure(figsize=(12, 6))  
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',  
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')  
plt.xlabel('K Value')  
plt.ylabel('Mean Error') 
