import pandas as pd
from sklearn import preprocessing

##################################
# Label Encoding
##################################

"""
Note:

- Fill missing values with "NONE" before fit_transform.
- .values converts dataframe type -> numpy array
- Label Encoding is great for tree based models like: 
1. Random Forest
2. Decision Trees
3. Extra Trees

Also for gradient boosting trees:
1. XGBoost
2. GBM
3. Lgbm

Important: This type of encoding is not suitable for NN, linear models or svm
simply because these  expect the input to be normalized or standardized.
"""

df = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv')

df.loc[:, 'ord_2'] = df.ord_2.fillna("NONE")

le = preprocessing.LabelEncoder()

df.loc[:, 'ord_2'] = le.fit_transform(df.ord_2.values)

####################################

"""
For linear models we can binarize each unique category in the column. 
For  eg: Freezing -> 2 -> 010.
Here for this feature column with 7 unique categories we can use 3 bit binary to represent each of them.
So 3 new cols for this one column.
A sparse matrix can used in this case but we only store values where matters i.e 1. Hence we dont need the entire mx3 sized matrix.
We can use a dictionary whose keys are the (x, y) coord for the places in the matrix where there is 1.
i.e (0, 2) -> 1
    (1, 0) -> 1
    
"""

# this is how to convert any numpy array as sparse CSR matrix
from scipy import sparse

ex_arr = np.array([
    [0, 0, 1], 
    [1, 0, 0],
    [1, 0, 1]
])

sparse_ex = sparse.csr_matrix(ex_arr)
print(f"Size of original example matrix: {ex_arr.nbytes}")
print(f"Size of the scipy CSR matrix: {sparse_ex.data.nbytes}")
# total size of csr matrix is still less  than the original one, each pos in matrix takes 8 bytes for int.
print(f"Total size of csr matrix: {sparse_ex.data.nbytes + sparse_ex.indptr.nbytes + sparse_ex.indices.nbytes}")

# the real difference is seen when you have thousands of exampples and thousands of features.
# remember a data of 1000 x 10_000 with only 5% ones takes up 8GB in the system while
# csr matrix took only 390 mb storage.
# There is one more way which takes even lesser space: one hot encoding
