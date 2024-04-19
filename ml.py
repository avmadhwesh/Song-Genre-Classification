import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

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

# ml supervised learning algorithm of multinomial naive bayes
mnb = MultinomialNB()
mnb.fit(xtrainVector, ytrain)
pred = mnb.predict(xtestVector)

# ml supervised learning algorithm of decision tree classifier
tree = DecisionTreeClassifier()
tree.fit(xtrainVector, ytrain)
treePred = tree.predict(xtestVector)

#ml supervised learning algorithm of support vector machine
svm = SVC()
svm.fit(xtrainVector, ytrain)
svmPred = svm.predict(xtestVector)

# Evaluate mnb model
accuracy = accuracy_score(ytest, pred)
print("Accuracy for MNB:", accuracy)

precision = precision_score(ytest, pred, average='weighted',zero_division=1)
recall = recall_score(ytest, pred, average='weighted')
f1 = f1_score(ytest, pred, average='weighted')

print("Precision for MNB:", precision)
print("Recall for MNB:", recall)
print("F1-score for MNB:", f1)

# Evaluate decision tree model
# Make predictions using Decision Tree Classifier
treePred = tree.predict(xtestVector)

# Evaluate Decision Tree model
accuracy_tree = accuracy_score(ytest, treePred)
precision_tree = precision_score(ytest, treePred, average='weighted', zero_division=1)
recall_tree = recall_score(ytest, treePred, average='weighted')
f1_tree = f1_score(ytest, treePred, average='weighted')

print("Accuracy for Decision Tree Classifier:", accuracy_tree)
print("Precision for Decision Tree Classifier:", precision_tree)
print("Recall for Decision Tree Classifier:", recall_tree)
print("F1-score for Decision Tree Classifier:", f1_tree)


# Evaluate svm model
# Make predictions using Support Vector Machine
svmPred = svm.predict(xtestVector)

# Evaluate Support Vector Machine model
accuracy_svm = accuracy_score(ytest, svmPred)
precision_svm = precision_score(ytest, svmPred, average='weighted', zero_division=1)
recall_svm = recall_score(ytest, svmPred, average='weighted')
f1_svm = f1_score(ytest, svmPred, average='weighted')

print("Accuracy for Support Vector Machine:", accuracy_svm)
print("Precision for Support Vector Machine:", precision_svm)
print("Recall for Support Vector Machine:", recall_svm)
print("F1-score for Support Vector Machine:", f1_svm)

##added

# Calculate confusion matrix 
confusion = confusion_matrix(ytest, pred)
genre_labels = list(mapping.keys())
plt.figure(figsize=(10, 8))
sns.heatmap(confusion, annot = True, fmt = 'd', cmap = 'Blues', xticklabels = genre_labels, yticklabels = genre_labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# confusion matrix for decision tree 
# Calculate confusion matrix for Decision Tree Classifier
confusion_tree = confusion_matrix(ytest, treePred)

# Visualize confusion matrix for Decision Tree Classifier
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_tree, annot=True, fmt='d', cmap='Blues', xticklabels=genre_labels, yticklabels=genre_labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Decision Tree Classifier')
plt.show()

#visualize for SVM
# Calculate confusion matrix for Support Vector Machine
confusion_svm = confusion_matrix(ytest, svmPred)

# Visualize confusion matrix for Support Vector Machine
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_svm, annot=True, fmt='d', cmap='Blues', xticklabels=genre_labels, yticklabels=genre_labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Support Vector Machine')
plt.show()

