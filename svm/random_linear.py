import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets


x=np.random.rand(100,2)
y=[]
#print(x)
j=0
while j<100 :
	
	if x[j][0]+x[j][1]<0.7:
		y.append(0)
	elif x[j][0]+x[j][1]<1.2:
		y.append(1)
	else:
		y.append(2)
	j=j+1


y = np.array(y)
print(x)
print(y)

svc = svm.SVC(kernel='rbf', C=1,gamma=0.1).fit(x, y)

print(svc)

x_min, x_max = 0.1 ,1
y_min, y_max = 0.1,1
h = (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))



plt.subplot(1, 1, 1) # to position in case of many figures in one frame
    
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Paired)

plt.xlim(xx.min(), xx.max())
plt.title('SVC with linear kernel')
plt.show()
