## 📘 Cross-Validation Tips

### Purpose

To select a cross-validation strategy that produces a **reliable estimate of generalization performance** while **avoiding data leakage** and **controlling computational cost**.

---

### 1. Strategy Selection Table

| Dataset characteristic                         | Dataset size / structure        | Recommended strategy                      | Exact procedure                                                                                  | Rationale                                              |
| ---------------------------------------------- | ------------------------------- | ----------------------------------------- | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------ |
| Standard classification (binary or multiclass) | Any size where CV is affordable | **Stratified K-Fold**                     | Split data into *k* folds preserving class proportions; train on *k−1* folds, validate on 1 fold | Ensures each fold reflects the true class distribution |
| Classification with severe class imbalance     | Any                             | **Stratified K-Fold**                     | Same as above; verify label distribution per fold                                                | Prevents folds dominated by a single class             |
| Very large classification dataset              | ≥ 1,000,000 samples             | **Hold-out validation**                   | Partition data into 90% training and 10% validation; train once, evaluate once                   | Full K-Fold is computationally prohibitive             |
| Regression with approximately uniform target   | Any                             | **K-Fold**                                | Split samples randomly into *k* folds; no stratification                                         | No discrete classes to preserve                        |
| Regression with skewed or heavy-tailed target  | Any                             | **Stratified K-Fold with target binning** | Bin target values, then apply Stratified K-Fold on bins                                          | Ensures target range is represented in all folds       |
| Small dataset                                  | ≤ 2,000 samples                 | **5–10 Fold Cross-Validation**            | Prefer higher *k* to maximize training data                                                      | Reduces variance due to limited data                   |
| Extremely small dataset                        | ≤ 500 samples                   | **Leave-One-Out (optional)**              | Train on *N−1* samples, validate on 1                                                            | Maximizes training data; high variance but low bias    |
| Time-ordered data (time series)                | Temporal dependency present     | **Temporal hold-out**                     | Train on earlier time period; validate on later period                                           | Prevents future information leakage                    |
| Repeated entities (users, patients, customers) | Any                             | **Group K-Fold**                          | Assign all samples of one entity to the same fold                                                | Prevents entity-level leakage                          |

---

### 2. Hold-Out Validation (Large Datasets)

**When to use**

* Dataset size makes repeated training impractical
* Model training or inference is computationally expensive

**Procedure**

* Partition data into *k* equal parts (commonly k = 10)
* Select **one part as validation**
* Use remaining parts for training
* All metrics are computed **only on the hold-out set**

**Example**

* Total samples: 1,000,000
* Hold-out size: 100,000
* Training size: 900,000

---

### 3. Time-Series Validation (Explicit Rule)

**Applicable when**

* Observations are indexed by time
* Future data must not influence past predictions

**Procedure**

* Training set: all observations from earlier time periods
* Validation set: a strictly later time period

**Example**

* Data available: 2015–2019
* Training period: 2015–2018
* Validation period: 2019

**Note**
Validation data must chronologically follow training data.

---

### 4. Stratified K-Fold for Regression (Sturges’ Rule)

Stratification is not directly defined for continuous targets.
To apply stratification, the target must first be discretized.

**Bin count selection (Sturges’ Rule)**

[
\text{Number of bins} = 1 + \log_2(N)
]

where
(N) = number of samples.

**Procedure**

1. Compute number of bins using Sturges’ Rule
2. Discretize the target into bins
3. Apply Stratified K-Fold using bin labels
4. Remove bins after fold assignment

**Use when**

* Target distribution is skewed
* Extreme target values are rare but important

---

### 5. Non-Negotiable Rules

| Rule                                                  | Explanation                  |
| ----------------------------------------------------- | ---------------------------- |
| Split data before feature engineering                 | Prevents leakage             |
| Validation data must represent deployment conditions  | Ensures meaningful metrics   |
| Same entity must never appear in train and validation | Prevents inflated scores     |
| Computational cost constrains CV choice               | More data ≠ more folds       |
| Correct CV matters more than model complexity         | Wrong CV invalidates results |

