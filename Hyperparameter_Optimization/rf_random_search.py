# rf_random_search.py
import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn import model_selection

if __name__ == "__main__":
    # Read the training data
    df = pd.read_csv("../input/mobile_train.csv")

    # X = features, y = target
    X = df.drop("price_range", axis=1).values
    y = df.price_range.values

    # Initialize Random Forest
    classifier = ensemble.RandomForestClassifier(n_jobs=-1)

    # Define the parameter distributions
    # Randomized search can also take distributions (like scipy.stats) 
    # instead of just fixed lists
    param_distributions = {
        "n_estimators": [100, 200, 250, 300, 400, 500],
        "max_depth": [1, 2, 5, 7, 11, 15],
        "criterion": ["gini", "entropy"]
    }

    # Initialize Randomized Search
    # n_iter=10 means it will randomly pick 10 combinations to test
    model = model_selection.RandomizedSearchCV(
        estimator=classifier,
        param_distributions=param_distributions,
        n_iter=10, 
        scoring="accuracy",
        verbose=10,
        n_jobs=1,
        cv=5
    )

    # Fit the model
    model.fit(X, y)

    # Output results
    print(f"Best score: {model.best_score_}")
    print("Best parameters set:")
    
    best_parameters = model.best_estimator_.get_params()
    for param_name in sorted(param_distributions.keys()):
        print(f"\t{param_name}: {best_parameters[param_name]}")
