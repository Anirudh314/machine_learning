from numpy import array
from sklearn.decomposition import PCA
# define a matrix
A = array([[1, 2], [3, 4], [5, 6]])
print(A)
# create the PCA instance
pca = PCA(2)
# fit on data
pca.fit(A)
# access values and vectors
print("pca components ",pca.components_) #eigen vector
print("pca variance ",pca.explained_variance_)
# transform data
B = pca.transform(A)
print("transformed",B)
