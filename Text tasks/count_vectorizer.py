from sklearn.feature_extraction.text import CountVectorizer
# create a corpus of sentences
corpus = [
 "hello, how are you?",
 "im getting bored at home. And you? What do you think?",
 "did you know about counts",
 "let's see if this works!",
 "YES!!!!"
]
# initialize CountVectorizer
ctv = CountVectorizer()
# fit the vectorizer on corpus
ctv.fit(corpus)
corpus_transformed = ctv.transform(corpus)
