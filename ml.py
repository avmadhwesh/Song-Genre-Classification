import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


#isolating the lyrics column and the genres column in the csv file and turning them into lists
dataF = pd.read_csv("newsong.csv")
lyrics = dataF["lyrics"].tolist()
genres = dataF["tag"].tolist()

#print(len(lyrics))
#print(len(genres))
'''
numRap = 0
numPop = 0
numRock = 0
numCountry = 0
numMisc = 0
for each in genres:
    if each == "rap":
        numRap += 1
    elif each == "pop":
        numPop += 1
    elif each == "rock":
        numRock += 1
    elif each == "country":
        numCountry += 1
    elif each == "misc":
        numMisc += 1
    
print(numRap)
print(numPop)
print(numRock)
print(numCountry)
print(numMisc)
'''

#creating an array of what different genres there are in the dataset
genreCategories = []
for genre in genres:
    if genre not in genreCategories:
        genreCategories.append(genre)

#preprocessing method: encoding each of the genres into a numerical value
ordinalEn = OrdinalEncoder()
genreCatList = []
for each in genreCategories:
    genreCatList.append([each])
encodedGenresCat = ordinalEn.fit_transform(genreCatList)
encodedGenresCat = encodedGenresCat.tolist()
mapping = {}
for i in range(len(genreCategories)):
    mapping[genreCategories[i]] = encodedGenresCat[i][0]
encodedGenres = []
for gen in genres:
    encodedGenres.append(mapping[gen])

#preprocessing method: splitting the data into train and test lists
xtrain, xtest, ytrain, ytest = train_test_split(lyrics, encodedGenres, test_size=0.2, random_state=42) 

#preprocessing method and feature extraction: counting the words in the lyrics and not looking at words such as "the", "is", etc.
onlyLetters = r'\b[a-zA-Z]+\b'
tfidf = TfidfVectorizer(stop_words="english", token_pattern=onlyLetters)
xtrainVector = tfidf.fit_transform(xtrain)
xtestVector = tfidf.transform(xtest)


#need to implement how pred and ytest differ - how accurate our model was at predicting the genre
# ml supervised learning algorithm of multinomial naive bayes
mnb = MultinomialNB()
mnb.fit(xtrainVector, ytrain)
pred = mnb.predict(xtestVector)

# Evaluate the model
accuracy = accuracy_score(ytest, pred)
print("Accuracy:", accuracy)

precision = precision_score(ytest, pred, average='weighted',zero_division=1)
recall = recall_score(ytest, pred, average='weighted')
f1 = f1_score(ytest, pred, average='weighted')

print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)




