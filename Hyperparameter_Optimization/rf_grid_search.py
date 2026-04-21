# rf_grid_search.py
import numpy as np
import pandas as pd
from sklearn import ensemble
# Note: although imported, metrics is often used via scoring strings in GridSearchCV
from sklearn import metrics 
from sklearn import model_selection

if __name__ == "__main__":
    # Read the training data
    # Ensure the path matches your local directory structure
    df = pd.read_csv("../input/mobile_train.csv")

    # Features are all columns except "price_range"
    X = df.drop("price_range", axis=1).values
    # Target variable
    y = df.price_range.values

    # Initialize the Random Forest Classifier
    # n_jobs=-1 uses all available CPU cores for the forest itself
    classifier = ensemble.RandomForestClassifier(n_jobs=-1)

    # Define the parameter grid for searching
    param_grid = {
        "n_estimators": [100, 200, 250, 300, 400, 500],
        "max_depth": [1, 2, 5, 7, 11, 15],
        "criterion": ["gini", "entropy"]
    }

    # Initialize Grid Search
    # cv=5 uses 5-fold cross-validation
    # n_jobs=1 in GridSearchCV because the RF itself is already using all cores
    model = model_selection.GridSearchCV(
        estimator=classifier,
        param_grid=param_grid,
        scoring="accuracy",
        verbose=10,
        n_jobs=1,
        cv=5
    )

    # Fit the model to the data
    model.fit(X, y)

    # Output the results
    print(f"Best score: {model.best_score_}")
    print("Best parameters set:")
    
    best_parameters = model.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        print(f"\t{param_name}: {best_parameters[param_name]}")
