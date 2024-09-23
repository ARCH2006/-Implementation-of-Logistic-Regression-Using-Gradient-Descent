# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

step1: start the program

step2: Use the standard libraries in python for finding linear regression.

step3:Set variables for assigning dataset values.

step4:Import linear regression from sklearn.

step5:Predict the values of array.

step6:Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

step7:Obtain the graph.

step8:End
## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by:212223040019
RegisterNumber: S.Archana
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv("C:/Users/ANANDAN S/Documents/ML labs/Placement_Data.csv")
dataset
#drapping the serial no and salary col
dataset = dataset.drop('sl_no', axis = 1)
dataset = dataset.drop('salary', axis = 1)
#categorical col for further labeling
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dataset["gender"] = dataset["gender"].astype('category')
dataset["ssc_b"] = dataset["ssc_b"].astype('category')
dataset["hsc_b"] = dataset["hsc_b"].astype('category')
dataset["hsc_s"] = dataset["hsc_s"].astype('category')
dataset["degree_t"] = dataset["degree_t"].astype('category')
dataset["workex"] = dataset["workex"].astype('category')
dataset["specialisation"] = dataset["specialisation"].astype('category')
dataset["status"] = dataset["status"].astype('category')
dataset.dtypes
dataset["gender"] = dataset["gender"].cat.codes
dataset["ssc_b"] = dataset["ssc_b"].cat.codes
dataset["hsc_b"] = dataset["hsc_b"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes
dataset["degree_t"] = dataset["degree_t"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["status"] = dataset["status"].cat.codes
dataset
x = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values
Y
#7.initialize the model parameters
import numpy as np
theta = np.random.randn(x.shape[1])
y = Y
#define sigmoid function 
def sigmoid(z):
    return 1/(1+np.exp(-z))
#define the loss function
def loss(theta,x,y):
    h = sigmoid(x.dot(theta))
    return -np.sum(y*np.log(h)+(1-y)*log(1-h))
#8. define the gradient descent algorithm
def gradient_descent(theta,x,y,alpha,num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(x.dot(theta))
        gradient = x.T.dot(h-y)/m
        theta = alpha*gradient
    return theta
#train the model
theta = gradient_descent(theta,x,y,alpha = 0.01,num_iterations = 1000)
#make predictions
def predict(theta,x):
    h = sigmoid(x.dot(theta))
    y_pred = np.where(h>=0.5,1,0)
    return y_pred
y_pred = predict(theta,x)
#evaluate themodel
accuracy = np.mean(y_pred.flatten()==y)
print('ACCURACY:', accuracy)
xnew = np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print(y_prednew)
xnew = np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print(y_prednew)


```

## Output:

![Screenshot 2024-09-19 084918](https://github.com/user-attachments/assets/433ae3ea-6f30-4896-b58d-c0f03befde74)
![Screenshot 2024-09-19 084905](https://github.com/user-attachments/assets/8c7c0ff2-75e5-4658-9797-85bfa0336fa2)
![Screenshot 2024-09-19 084849](https://github.com/user-attachments/assets/4e8cd986-13df-435c-8d90-2fc158400ced)
![Screenshot 2024-09-19 084929](https://github.com/user-attachments/assets/ca99528d-00c2-4317-a5b7-d92be61de411)

## Result:
  The logistic regression is implemented successfully using gradient Descent

