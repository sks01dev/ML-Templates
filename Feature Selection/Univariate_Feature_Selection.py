from sklearn.feature_selection import (
    chi2,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
    SelectKBest,
    SelectPercentile,
)


class UnivariateFeatureSelection:
    def __init__(self, n_features, problem_type, scoring):
        """
        Custom univariate feature selection wrapper on
        different univariate feature selection models from scikit-learn.

        Parameters:
        ----------
        n_features : int or float
            If int -> SelectKBest
            If float -> SelectPercentile (0.0 to 1.0)

        problem_type : str
            "classification" or "regression"

        scoring : str
            Scoring function name
        """

        if problem_type == "classification":
            valid_scoring = {
                "f_classif": f_classif,
                "chi2": chi2,
                "mutual_info_classif": mutual_info_classif,
            }
        elif problem_type == "regression":
            valid_scoring = {
                "f_regression": f_regression,
                "mutual_info_regression": mutual_info_regression,
            }
        else:
            raise ValueError("Invalid problem type")

        if scoring not in valid_scoring:
            raise ValueError("Invalid scoring function")

        if isinstance(n_features, int):
            self.selection = SelectKBest(
                score_func=valid_scoring[scoring],
                k=n_features,
            )
        elif isinstance(n_features, float):
            self.selection = SelectPercentile(
                score_func=valid_scoring[scoring],
                percentile=int(n_features * 100),
            )
        else:
            raise ValueError("n_features must be int or float")

    def fit(self, X, y):
        return self.selection.fit(X, y)

    def transform(self, X):
        return self.selection.transform(X)

    def fit_transform(self, X, y):
        return self.selection.fit_transform(X, y)



## Sample Use
"""
ufs = UnivariateFeatureSelction( 
n_features=0.1,  
problem_type="regression",  
scoring="f_regression" 
) 
ufs.fit(X, y) 
X_transformed = ufs.transform(X) 
"""
