# Group 48 Project 
## Introduction/Background
### Literature Review
Music information retrieval, or MIR, is a growing science where datasets are used to extract musical information from songs or other forms of music. [9] There has been a growing interest in utilizing song lyrics as a valuable source of information for genre classification. Existing studies have explored various methods, such as natural language processing and statistical models, to extract meaningful features from lyrics and predict musical genres accurately [10]. 

### Dataset
The dataset encompasses a wide range of musical genres. The musiXmatch dataset offers word frequency information in a bag-of-words representation which captures linguistic patterns, which genre-correlated patterns can be discerned from.The inclusion of various genres ensures the model's capability to generalize across different musical styles.

Dataset Link: http://millionsongdataset.com/musixmatch/ 

## Problem Definition
### Problem
Our objective is to develop a machine learning model capable of accurately predicting the genre of a song from its lyrics. This process involves: determining a set of appropriate music genres, converting song lyrics into usable data, and classifying lyrics into distinct musical genres by leveraging patterns extracted from the content of lyrics.

### Motivation
Our motivation is to contribute to the scope of streaming platforms and other services. Automated genre classification is crucial to the function of platforms and recommendations systems. By making a model that can place songs into genres without platform manager intervention, the user experience on streaming platforms may be enhanced. Further, classifying the lyric-genre could provide insights into cultural traditions and the evolving trends within different music genres.

## Methods
The three data preprocessing methods we plan on utilizing are feature extraction, feature selection, and data splitting. Feature extraction will allow us to process the raw lyric data into underlying patterns found between lyrics and genres. Feature selection will enable us to pick out the most relevant features to focus on. Data splitting will allow us to split our data into a train set and a test set [3]. 
The ML algorithms and supervised learning methods we plan to use are Support Vector Machine Classification, Decision Tree Classification, and Multinomial Naive Bayes. These methods are useful in classifying the features found into categories based on different parameters [1, 2]. For example, Multinomial Naive Bayes will be useful in predicting the genre based on the frequency of certain words [2]. 
All of these methods can be found in the scikit-learn library.

#### Midterm Progress: 
The three data preprocessing methods we implemented were encoding each of the genres into numbers, feature extraction, and splitting our dataset into a train and a test set. In order to encode each of the genres into numerical values, we used sklearn implementation of Ordinal Encoder. We decided to encode our genres because they were originally all strings, but as numbers it will be easier to compare our predicted labels from our trained models against the true labels when measuring the accuracy of our models. Another preprocessing method we implemented was feature extraction. Since lyrics can have numbers, special characters, and insignificant words such as “the”, “is”, “of”, etc., we decided to get rid of these words in our lyrics dataset, extracting only the essential words of the lyrics to train our models with. Lastly, we split our dataset into a training dataset and a test dataset. Since we are implementing supervised learning ML algorithms, it is important to split the data so we can train the model with the train set and see if it accurately predicts the true labels of the test set. 


## Results/Discussion
We plan on using 4 quantitative metrics. The first metric is accuracy [4,5]. The second metric is precision [4,6]. The third metric is recall [4,7]. The last metric is the F1 score [4,8]. The goal for all would be to maximize the value as close to 1 as possible, but a value of .8 or above can be considered successful for accuracy and values above .7 for each genre can be considered successful for precision, recall and F1 score [5,6,7,8]. We expect to get an accuracy score above .8 and precision, recall and F1 scores above .7 for each class. 

#### Midterm Results:
- add actual results
As we move forward, we hope to train the machine learning model with the data continuously so that it learns the dataset well and is able to predict as we apply it forward. We will also see if there are any errors that are appearing in the processed data and results. This will give us an idea on how to further improve the testing/training set to remove any faulty data points. We do not want the algorithm to be based on incorrect data.  
Further, we still have to extend this process to the other 2 models since it has only been implemented on one. We need to train the model using the new dataset and the appropriate features, validate the model to ensure it’s performing as expected, and document the whole process. We need to use quantitative metrics to evaluate their performance and analyze the results. From there, we will hopefully be able to successfully predict song genre based off lyrics.


## References
[1] “1.4. Support Vector Machines — scikit-learn 0.22.1 documentation,” scikit-learn.org. https://scikit-learn.org/stable/modules/svm.html#classification <br>
[2] “1.9. Naive Bayes — scikit-learn 0.23.2 documentation,” scikit-learn.org. https://scikit-learn.org/stable/modules/naive_bayes.html#multinomial-naive-bayes <br>
[3] scikit-learn, “sklearn.model_selection.train_test_split — scikit-learn 0.20.3 documentation,” Scikit-learn.org, 2018. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html <br>
[4] “3.3. Metrics and scoring: quantifying the quality of predictions — scikit-learn 0.24.1 documentation,” scikit-learn.org. https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics <br>
[5] “sklearn.metrics.accuracy_score — scikit-learn 0.24.1 documentation,” scikit-learn.org. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score <br>
[6] “sklearn.metrics.precision_score — scikit-learn 0.24.1 documentation,” scikit-learn.org. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score <br>
[7] “sklearn.metrics.recall_score,” scikit-learn. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score <br>
[8] “sklearn.metrics.f1_score,” scikit-learn. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score <br>
[9]  Li, T., Ogihara, M., & Tzanetakis, G. (Eds.). (2011). Music data mining. CRC Press. <br>
[10] Boonyanit, Anna & Dahl, Andrea. Music Genre Classification using Song Lyrics. Stanford University.
[11] Nayak, Nikhil, Apr 8, 2022, "Genius Song Lyrics", Kaggle. [Online]. Available: https://www.kaggle.com/datasets/carlosgdcj/genius-song-lyrics-with-language-information

## Gantt Chart
![mlgantt_1](https://github.gatech.edu/storage/user/70451/files/3bcd8922-2d1d-4232-adce-ce0541a0e797)
![mlgantt_2](https://github.gatech.edu/storage/user/70451/files/6359d49c-dfab-46a5-9c07-5a43acd085da)
![mlgantt_3](https://github.gatech.edu/storage/user/70451/files/b2adf859-a980-450a-b2a4-37c987700c45)

## Contribution Table
![mlcontribution](https://github.gatech.edu/storage/user/70451/files/7c7df4a3-1266-487f-8964-bd1e48ad75e8)
