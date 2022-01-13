import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
import nltk
from nltk.corpus import stopwords
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv("tweet_emotions.csv")
#
# for index, row in data.iterrows():
#     if row["sentiment"] == ""

x = data["content"]
y = data["sentiment"]
# y_new = []

# for count, i in enumerate(y):
#     if i == "empty":
#         y_new.append(0)
#     elif i == "sadness":
#         y_new.append(1)
#     elif i == "enthusiasm":
#         y_new.append(2)
#     elif i == "neutral":
#         y_new.append(3)
#     elif i == "worry":
#         y_new.append(4)
#     elif i == "love":
#         y_new.append(5)
#     elif i == "fun":
#         y_new.append(6)
#     elif i == "hate":
#         y_new.append(7)

# y = y_new
#
vectorizer = HashingVectorizer(n_features=12)
x = vectorizer.transform(x).toarray()


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)

np.save("x_train.npy", x_train)
np.save("y_train.npy", y_train)
np.save("x_test.npy", x_test)
np.save("y_test.npy", y_test)