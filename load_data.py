import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer

data = pd.read_csv("tweet_emotions.csv")

x = data["content"]
y = data["sentiment"]


vectorizer = HashingVectorizer(n_features=12)
x = vectorizer.transform(x).toarray()

print(x[0])