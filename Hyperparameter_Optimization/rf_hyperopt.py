# rf_hyperopt.py
import numpy as np
import pandas as pd
from functools import partial
from sklearn import ensemble, metrics, model_selection
from hyperopt import hp, fmin, tpe, Trials
from hyperopt.pyll.base import scope

def optimize(params, x, y):
    """
    Main optimization function for Hyperopt.
    """
    # Initialize model with current parameters from the search space
    model = ensemble.RandomForestClassifier(**params, n_jobs=-1)

    # Cross-validation
    kf = model_selection.StratifiedKFold(n_splits=5)
    accuracies = []

    for train_idx, test_idx in kf.split(X=x, y=y):
        xtrain, xtest = x[train_idx], x[test_idx]
        ytrain, ytest = y[train_idx], y[test_idx]

        model.fit(xtrain, ytrain)
        preds = model.predict(xtest)
        fold_accuracy = metrics.accuracy_score(ytest, preds)
        accuracies.append(fold_accuracy)

    # Return negative accuracy (Hyperopt minimizes)
    return -1 * np.mean(accuracies)

if __name__ == "__main__":
    df = pd.read_csv("../input/mobile_train.csv")
    X = df.drop("price_range", axis=1).values
    y = df.price_range.values

    # Define the parameter space using Hyperopt syntax
    param_space = {
        # scope.int converts the float output of quniform to an integer
        "max_depth": scope.int(hp.quniform("max_depth", 1, 15, 1)),
        "n_estimators": scope.int(hp.quniform("n_estimators", 100, 1500, 1)),
        "criterion": hp.choice("criterion", ["gini", "entropy"]),
        "max_features": hp.uniform("max_features", 0.01, 1)
    }

    # Use partial to freeze X and y into the function
    optimization_function = partial(optimize, x=X, y=y)

    # Initialize Trials object to track progress
    trials = Trials()

    # Run the optimization
    # tpe.suggest is the search algorithm
    hopt = fmin(
        fn=optimization_function,
        space=param_space,
        algo=tpe.suggest,
        max_evals=15,
        trials=trials
    )

    # Note: hopt returns indices for hp.choice, not the value itself
    print("Best parameters found:")
    print(hopt)
