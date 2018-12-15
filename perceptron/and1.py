from random import choice
from numpy import array, dot, random
import numpy as np
import matplotlib.pyplot as plt


def step_fun(x):
	if(x>1):
		return 1
	else:
		return 0

training_data = [(array([0,0,1]), 0),(array([0,1,1]), 0),(array([1,0,1]), 0),(array([1,1,1]), 1),]

w = random.rand(3)
print(w)

errors = []
learning_rate = 0.05 #magnitude of jump to reach minima
n = 100 


for i in range(n):
    x, expected = choice(training_data) 
    result = dot(w,x)
    #print(result)
    err =expected - step_fun(result)
    errors.append(err)
    w=w+learning_rate*err*x

c=0
for i in range(n):
	x,expected = choice(training_data) 
	result=dot(w,x)
	err = expected-step_fun(result)
	if(err!=0):
		c=c+1
		
print("accuracy = "  ,(n-c)*100/n )

x_arr = np.arange(0,1,0.01)
y_arr = np.arange(0,1,0.01)

print(x_arr)

x1,y1 = [],[]
x2,y2 = [],[]

for xi in x_arr:
	for yi in y_arr:
		pi = dot(w, [xi,yi,1])
		if int(pi) == 1:
			x1.append(xi), y1.append(yi) 		
		else:
			x2.append(xi), y2.append(yi) 		

plt.scatter(x1,y1,color='grey')

plt.scatter(x2,y2,color='cyan')

x1,y1 = [],[]
x2,y2 = [],[]

for xi,di in training_data:
	if di==1:
		x1.append(xi[0]), y1.append(xi[1])
	else:
		x2.append(xi[0]), y2.append(xi[1])

plt.scatter(x1,y1,color='blue')

plt.scatter(x2,y2,color='red')

plt.show()
c=0
for xi,di in training_data:
	pi = dot(w, xi)
	print(pi,di,x)
    
	if int(pi)==di:
		c+=1
print(c)



