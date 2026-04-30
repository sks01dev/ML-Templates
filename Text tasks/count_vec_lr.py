import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer

# Required if running for the first time
nltk.download('punkt')

if __name__ == "__main__":
    # 1. Load Data
    # Ensure the path matches your local setup
    df = pd.read_csv("../input/imdb.csv")

    # 2. Preprocessing
    # Map 'positive' to 1 and 'negative' to 0
    df.sentiment = df.sentiment.apply(
        lambda x: 1 if x == "positive" else 0
    )

    # Create kfold column and shuffle data
    df["kfold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)

    # 3. K-Fold Setup
    y = df.sentiment.values
    kf = model_selection.StratifiedKFold(n_splits=5)

    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f

    # 4. Training Loop
    for fold_ in range(5):
        # Split train and validation
        train_df = df[df.kfold != fold_].reset_index(drop=True)
        test_df = df[df.kfold == fold_].reset_index(drop=True)

        # Initialize CountVectorizer
        # token_pattern=None is required when using a custom tokenizer
        count_vec = CountVectorizer(
            tokenizer=word_tokenize,
            token_pattern=None
        )

        # Fit and Transform
        count_vec.fit(train_df.review)
        xtrain = count_vec.transform(train_df.review)
        xtest = count_vec.transform(test_df.review)

        # Initialize Logistic Regression
        # Note: increased max_iter to avoid "Failed to converge" errors
        model = linear_model.LogisticRegression(max_iter=1000)

        # Fit and Predict
        model.fit(xtrain, train_df.sentiment)
        preds = model.predict(xtest)

        # 5. Metrics
        accuracy = metrics.accuracy_score(test_df.sentiment, preds)
        
