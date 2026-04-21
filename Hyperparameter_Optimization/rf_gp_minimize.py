# rf_gp_minimize.py
import numpy as np
import pandas as pd
from functools import partial
from sklearn import ensemble, metrics, model_selection
from skopt import gp_minimize, space

def optimize(params, param_names, x, y):
    """
    Main optimization function for gp_minimize.
    Returns negative accuracy because gp_minimize only minimizes.
    """
    # Map parameter names to the values suggested by gp_minimize
    params = dict(zip(param_names, params))

    # Initialize model with suggested parameters
    model = ensemble.RandomForestClassifier(**params, n_jobs=-1)

    # Use Stratified K-Fold for stable cross-validation
    kf = model_selection.StratifiedKFold(n_splits=5)
    accuracies = []

    for train_idx, test_idx in kf.split(X=x, y=y):
        xtrain, xtest = x[train_idx], x[test_idx]
        ytrain, ytest = y[train_idx], y[test_idx]

        model.fit(xtrain, ytrain)
        preds = model.predict(xtest)
        
        fold_accuracy = metrics.accuracy_score(ytest, preds)
        accuracies.append(fold_accuracy)

    # Return negative mean accuracy
    return -1 * np.mean(accuracies)

if __name__ == "__main__":
    df = pd.read_csv("../input/mobile_train.csv")
    X = df.drop("price_range", axis=1).values
    y = df.price_range.values

    # Define the search space
    param_space = [
        space.Integer(3, 15, name="max_depth"),
        space.Integer(100, 1500, name="n_estimators"),
        space.Categorical(["gini", "entropy"], name="criterion"),
        space.Real(0.01, 1, prior="uniform", name="max_features")
    ]

    param_names = ["max_depth", "n_estimators", "criterion", "max_features"]

    # Wrap the function to fix X and y
    optimization_function = partial(optimize, param_names=param_names, x=X, y=y)

    # Run Bayesian Optimization
    result = gp_minimize(
        optimization_function,
        dimensions=param_space,
        n_calls=15,           # Total iterations
        n_random_starts=10,   # Initial random explorations
        verbose=True
    )

    # Print best results
    best_params = dict(zip(param_names, result.x))
    print(f"Best Accuracy: {-1 * result.fun}")
    print(f"Best Parameters: {best_params}")
