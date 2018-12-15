#program to show importance of each feature 


import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn import datasets


iris = datasets.load_iris()
x = iris.data[:, :4]
#print(x)


x=scale(x) #scale used to make various feature be in same scale 

pca= PCA(n_components=4)

pca.fit(x)


#The amount of variance that each PC explains
var= pca.explained_variance_ratio_



print(var)
print("importance of each feature",np.round(pca.explained_variance_ratio_, decimals=4)*100)


#print("\n\n----------")
#Cumulative Variance explains
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

print ("cumulative importance",var1)


#plt.plot()  #cumulative importance of each feature 

#plt.show()


