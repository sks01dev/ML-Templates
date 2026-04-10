import pandas as pd
from sklearn import linear_model, metrics
from sklearn.datasets import make_classification


class GreedyFeatureSelection:
    """
    A simple and custom class for greedy feature selection.
    You may need to adapt it depending on your dataset.
    """

    def evaluate_score(self, X, y):
        """
        Evaluate model performance using AUC (overfitted).

        NOTE:
        This evaluates on the same data used for training (overfitting).
        For a more correct approach, use cross-validation (OOF AUC).

        Parameters:
        ----------
        X : array-like
            Training data
        y : array-like
            Target labels

        Returns:
        -------
        float
            AUC score
        """
        model = linear_model.LogisticRegression(max_iter=1000)
        model.fit(X, y)

        predictions = model.predict_proba(X)[:, 1]
        auc = metrics.roc_auc_score(y, predictions)
        return auc

    def _feature_selection(self, X, y):
        """
        Perform greedy feature selection.

        Parameters:
        ----------
        X : numpy array
        y : numpy array

        Returns:
        -------
        tuple: (best_scores, selected_features)
        """
        good_features = []
        best_scores = []

        num_features = X.shape[1]

        while True:
            this_feature = None
            best_score = 0

            for feature in range(num_features):
                if feature in good_features:
                    continue

                selected_features = good_features + [feature]
                xtrain = X[:, selected_features]

                score = self.evaluate_score(xtrain, y)

                if score > best_score:
                    this_feature = feature
                    best_score = score

            # If a feature improved the score, keep it
            if this_feature is not None:
                good_features.append(this_feature)
                best_scores.append(best_score)

            # Stop if no improvement
            if len(best_scores) > 1:
                if best_scores[-1] <= best_scores[-2]:
                    break
            else:
                # If no feature was added at all, break
                if this_feature is None:
                    break

        return best_scores[:-1], good_features[:-1]

    def __call__(self, X, y):
        """
        Run feature selection and transform dataset.
        """
        scores, features = self._feature_selection(X, y)
        return X[:, features], scores


if __name__ == "__main__":
    # Generate binary classification data
    X, y = make_classification(n_samples=1000, n_features=100)

    # Apply greedy feature selection
    selector = GreedyFeatureSelection()
    X_transformed, scores = selector(X, y)

    print("Selected feature count:", X_transformed.shape[1])
    print("Scores:", scores)
