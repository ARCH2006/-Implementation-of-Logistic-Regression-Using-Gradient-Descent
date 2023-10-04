# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Use the standard libraries in python for finding linear regression.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Predict the values of array.

5.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

6.Obtain the graph.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by:212222230118
RegisterNumber: S.Renuga

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("ex2data1.txt",delimiter=',')
X=data[:,[0,1]]
y=data[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label="Admitted")
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label="Not admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
  return 1 / (1 + np.exp(-z))

plt.plot()
X_plot = np.linspace(-10, 10, 100)
plt.plot(X_plot, sigmoid(X_plot))
plt.show()

def costFunction(theta,X,y):
  h = sigmoid(np.dot(X, theta))
  J = -(np.dot(y, np.log(h)) + np.dot(1 - y, np.log(1-h))) / X.shape[0]
  grad = np.dot(X.T, h-y) / X.shape[0]
  return J,grad

X_train = np.hstack((np.ones((X.shape[0], 1)), X))
theta = np.array([0, 0, 0])
J, grad = costFunction(theta, X_train, y)
print(J)
print(grad)

X_train = np.hstack((np.ones((X.shape[0], 1)), X))
theta = np.array([-24, 0.2, 0.2])
J, grad = costFunction(theta, X_train, y)
print(J)
print(grad)

def cost(theta, X, y):
  h= sigmoid(np.dot(X, theta))
  J = - (np.dot(y, np.log(h)) + np.dot(1 - y, np.log(1 - h))) / X.shape[0]
  return J

def gradient(theta, X, y):
  h = sigmoid(np.dot(X, theta))
  grad = np.dot(X.T, h - y) / X.shape[0]
  return grad

X_train = np.hstack((np.ones((X.shape[0], 1)), X))
theta = np.array([0, 0, 0])
res = optimize.minimize(fun=cost, x0=theta, args=(X_train, y),
                        method='Newton-CG', jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta, X, y):
  x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
  y_min, y_max = X[:, 0].min() - 1, X[:, 0].max() + 1
  xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                       np.arange(y_min, y_max, 0.1))
  X_plot = np.c_[xx.ravel(), yy.ravel()]
  X_plot = np.hstack((np.ones((X_plot.shape[0], 1)), X_plot))
  y_plot = np.dot(X_plot, theta).reshape(xx.shape)

  plt.figure()
  plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label="Admitted")
  plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label="Not admitted")
  plt.contour(xx, yy, y_plot, levels=[0])
  plt.xlabel("Exam 1 score")
  plt.ylabel("Exam 2 score")
  plt.legend()
  plt.show()

plotDecisionBoundary(res.x, X, y)

prob = sigmoid(np.dot(np.array([1, 45, 85]), res.x))
print(prob)

def predict(theta, X):
  X_train = np.hstack((np.ones((X.shape[0], 1)), X))
  prob = sigmoid(np.dot(X_train, theta))
  return (prob >= 0.5).astype(int)

n(predict(resnp.mea.x, X) == y)

*/
```

## Output:

## ARRAY VALUE OF X:

![Screenshot 2023-09-25 141211](https://github.com/RENUGASARAVANAN/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119292258/2675fc59-95d8-4d86-bb5b-8fb302978754)

## ARRAY VALUE OF Y:

![Screenshot 2023-09-25 141312](https://github.com/RENUGASARAVANAN/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119292258/decddb30-a4ae-4219-9c70-bc57ee462371)

## EXAM 1 SCORE GRAPH:

![Screenshot 2023-09-25 141348](https://github.com/RENUGASARAVANAN/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119292258/c8ae4f74-287e-4889-a2ed-d13bfbe45df6)

## SIGMOID FUNCTION GRAPH:

![Screenshot 2023-09-25 141439](https://github.com/RENUGASARAVANAN/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119292258/a62c56cf-fddc-42ae-8ff6-486888b43c26)

## X_TRAIN_GRAD VALUE:

![Screenshot 2023-09-25 141513](https://github.com/RENUGASARAVANAN/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119292258/c01774cf-3fcd-4762-8723-cc7aefb9d568)

## Y_TRAIN_GRAD VALUE:

![Screenshot 2023-09-25 141543](https://github.com/RENUGASARAVANAN/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119292258/56cc8727-6ed4-4a6e-912e-ee286c9cad41)

## PRINT RES.X:

![Screenshot 2023-09-25 141618](https://github.com/RENUGASARAVANAN/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119292258/5ee84825-f907-4018-8982-f34257609776)

## DECISION BOUNDARY_GRAPH FOR EXAM SCORE:

![Screenshot 2023-09-25 141704](https://github.com/RENUGASARAVANAN/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119292258/f58e3fef-b3f3-4366-bd97-87553cdbf72b)

## PROBABILITY VALUE:

![Screenshot 2023-09-25 141743](https://github.com/RENUGASARAVANAN/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119292258/d4b8238c-561b-48e7-81d0-2ed288bb03b4)

## PREDICTION VALUE OF MEAN:

![Screenshot 2023-09-25 141821](https://github.com/RENUGASARAVANAN/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119292258/75a8b857-eeeb-45b4-843b-c94c5074bb55)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

