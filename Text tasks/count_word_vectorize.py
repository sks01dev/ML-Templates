from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
# create a corpus of sentences
corpus = [
 "hello, how are you?",
 "im getting bored at home. And you? What do you think?",
 "did you know about counts",
 "let's see if this works!",
 "YES!!!!"
]
# initialize CountVectorizer with word_tokenize from nltk
# as the tokenizer
ctv = CountVectorizer(tokenizer=word_tokenize, token_pattern=None)
# fit the vectorizer on corpus
ctv.fit(corpus)
corpus_transformed = ctv.transform(corpus)
print(ctv.vocabulary_)

# Output
{'hello': 14, ',': 2, 'how': 16, 'are': 7, 'you': 27, '?': 4, 'im': 18,
'getting': 13, 'bored': 9, 'at': 8, 'home': 15, '.': 3, 'and': 6, 'what':
24, 'do': 12, 'think': 22, 'did': 11, 'know': 19, 'about': 5, 'counts':
10, 'let': 20, "'s": 1, 'see': 21, 'if': 17, 'this': 23, 'works': 25,
'!': 0, 'yes': 26}
