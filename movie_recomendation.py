import numpy as np
import pandas as pd
import difflib
#difflib library is used to find closest match to what user enters, as user can enter 'movie name' with spelling mistakes
from sklearn.feature_extraction.text import TfidfVectorizer
#this library helps in converting textual data to feature vector
from sklearn.metrics.pairwise import cosine_similarity
#this library helps in giving similarity confidence score for the movies; the movies which have highest similarity score are most similar

#loading data from 'movies.csv'
movies_data=pd.read_csv("movies.csv")
#print first 5 rows:
#print(movies_data.head())

#to find number of rows and columns:
#print(movies_data.shape)

#using only particular features:
selected_features=['genres','keywords','tagline','cast','director']

#filling null values with null string:
for i in selected_features:
    movies_data[i]=movies_data[i].fillna('')

#combining only selected features into another data variable:
combined_features=movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']
#print(combined_features)

#converting text data to feature vector i.e numerical values:
vectorizer=TfidfVectorizer()
feature_vectors=vectorizer.fit_transform(combined_features)

#print(feature_vectors)

#getting the similarity scores using cosine_similarity:
similarity=cosine_similarity(feature_vectors)


#getting user input:
movie_name=input("enter your favourite movie name")

#checking if the movie is present in our data set and finding closest matches:
all_movies=movies_data['title'].tolist() #this will contain the names of all the movies in our data set
find_match=difflib.get_close_matches(movie_name,all_movies) #find match is a list containing closest movie name match results
#print(find_match)

close_match=find_match[0]

#finding index of close_match
ind=-1
flag=0
for i in range(len(all_movies)):
    if(all_movies[i]==close_match):
        ind=i
        flag=1
        #print(i)
        break

if flag==0:
    print("no match found")
else:
    #getting similarity scores for movie entered by user:
    similarity_scores=[]
    for i in range(len(similarity[ind])):
        similarity_scores.append((i,similarity[ind][i]))

    #print(similarity_scores)
    #print(len(similarity_scores))

    #since similarity_scores is a list with each element as tupple (index,similarity-score) therefore sorting based on similarity-score:
    sorted_similar_movies=sorted(similarity_scores,key=lambda x:x[1],reverse=True)
    #print(sorted_similar_movies)

    #giving user top 10 recomendations:
    print("top 10 recomendations are :")
    #the first movie after sorting ,that is the movie with the highest similarity rating will be the movie entered by user itself
    #so we start index from 1 not 0 in below for loop to get different movies for the user
    for i in range(1,11):
        print(all_movies[sorted_similar_movies[i][0]])












