from numpy import mean
from numpy import cov,array
from numpy.linalg import eig
import numpy as np
# define a matrix
A = array([[1, 2], [3, 4], [5, 6]])
print("array",A)
# calculate the mean of each column
M = mean(A.T, axis=1)

print("mean",M)
# center columns by subtracting column means
C = A - M
print("center",C)
# calculate covariance matrix of centered matrix
V = cov(C.T)
print("covariance matrix",V)
# eigendecomposition of covariance matrix
values, vectors = eig(V)
print("eigen vector",vectors)
print("eigen value ",values)
# project data
P = vectors.T.dot(C.T)
print("projected data ",P.T)
