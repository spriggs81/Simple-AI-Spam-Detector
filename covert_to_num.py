from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from training_data import data

#Separate texts and labels
texts = [item[0] for item in data]
labels = [item[1] for item in data]

#Initialize the CountVectorizer
vectorizer = CountVectorizer()

#Fit the vectorizer to our texts & transform them into numerical features
X = vectorizer.fit_transform(texts)

#Convert labels to numerical format (0 for 'not spam' & 1 for 'spam')
label_mapping = {"not spam": 0, "spam": 1}

#using pandas for easy mapping
#'y' will be the target labels (0s & 1s)
y = pd.Series(labels).map(label_mapping).values

def test_data(): 
    print("Original Texts (first 2):", texts[:2])
    print("Transformed Features (first 2 rows of X as dense array):\n",X[:2] ,"\n\n",X.toarray()[:2]) # .toarray() converts sparse matrix to dense for viewing
    print("Numerical Labels (y):", y)
    print("Vocabulary learned by vectorizer:", vectorizer.get_feature_names_out())