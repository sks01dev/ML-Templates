import numpy as np
import pandas as pd
from sklearn.model_selection import ( KFold, StratifiedKFold, StratifiedGroupKFold )

# =========================================================
# 1. STRATIFIED K-FOLD (Binary / Multiclass Classification)
# =========================================================

def skfold(df, target_col, n_splits=5):
    """
    Use this for:
    - Binary classification
    - Multiclass classification
    - Imbalanced classes

    Creates a new column with 0-based indexing indicating 'fold'.
    """
    df = df.copy()
    df['fold'] = -1

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df[target_col])):
        df.loc[val_idx, 'fold'] = fold

    return df


# ==========================================
# 2. SIMPLE K-FOLD (Regression)
# ==========================================

def kf_regression(df, target_col, n_splits=5):
    """
    Use this for:
    - Regression tasks, without skewness
    """

    df = df.copy()
    df['fold'] = - 1

    kf = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        df.loc[val_idx, 'fold'] = fold

    return df

# ==================================================
# 3. STRATIFIED K-FOLD FOR REGRESSION (BINNING)
# ==================================================

def skf_regression(df, target_col, n_splits=5):
    """
    Use this for skewed target distribution in regression problems.
    Here, we use the Sturge's Formula to decide the number of bins.
    """

    df = df.copy()
    df['fold'] = - 1

    num_bins = int(1 + np.floor(np.log2(len(df)))) # Sturge's Rule
    df['bins'] = pd.cut(df[target_col], bins=num_bins, labels=False)

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['bins'])):
        df.loc[val_idx, 'fold'] = fold

    # drop the bins column
    df.drop('bins', axis=1, inplace=True)
    return df


# ==========================================
# 4. GROUP K-FOLD (User / Patient / Customer)
# ==========================================

def group_kf(df, group_col, n_splits=5):
    """
    Use this when:
    - Same user/patient/ids appear multiple times
    - To prevent leakage
    """

    df = df.copy()
    df['fold'] = -1

    gkf = GroupKFold(n_splits=n_splits)

    for fold, (trn_idx, val_idx) in enumerate(gkf.split(df,groups=df[group_col])):
        df.loc[val_idx, 'fold'] = fold

    return df



