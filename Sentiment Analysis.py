#importing the modules

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing the dataset

dataset=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)
#While implementing NLP we remove the double quotes from the text to avoid parsing errors
#to read tsv file we add the delimiter and set it to '\t'
#we set quoting to 3 to avoid the double quotes

#Cleaning the texts

import re 
import nltk
nltk.download('stopwords') # it helps to remove the unusable words like 'the','and','er' etc
from nltk.corpus import stopwords #it imports the stopwords and the previous line downloads the stopwords
from nltk.stem.porter import PorterStemmer  # For stemming the reviews, like "I loved this restaurant" and if we apply stemming on 'loved' then it can be changed to 'lvoe'
corpus=[]
for i in range(0,1000):
    review=re.sub('[^a-zA-Z]',' ',dataset['Review'][i]) # we are removing all the punctuation marks with a space so we know the different words
    review=review.lower() #transforming all upper case letter to lower case letter
    review=review.split() # Splitting the elements into words to apply stemming
    ps=PorterStemmer()
    all_stopwords=stopwords.words('english')
    all_stopwords.remove('not') # to remove not from the stopwords
    review = [ps.stem(word)for word in review if not word in set(all_stopwords)]  # running a loop through each review and only considering those words that are not stopwords and then stemming each of that word
    review = ' '.join(review)
    corpus.append(review)

print(corpus)

#Creating the Bag of words model

from sklearn.feature_extraction.text import CountVectorizer #count vectorizer basicallty removes the more useless words like 'texture' ,'steve' etc
cv=CountVectorizer(max_features=1500) #we put 1500 because after running the code without the parameter it gave 1566 and we put 1500 so we have a matrix of 1500 useful words after tokenisation
x = cv.fit_transform(corpus).toarray()  # we add to array for making it a 2d array
y=dataset.iloc[:,-1].values

#Splitting the dataset into training set and test set

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=1)

#Training the Random FOrest model on the training set

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators =100,criterion='entropy',random_state=0)
classifier.fit(x_train,y_train)

#Predict the test set results

y_pred=classifier.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))

#Making a confusion matrix 

from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test,y_pred)
print(cm)

score=accuracy_score(y_test,y_pred)
print(score)