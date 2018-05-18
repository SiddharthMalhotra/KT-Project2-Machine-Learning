# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 21:46:14 2017

@author: Siddharth Malhotra
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img

d  = np.loadtxt('train.arff.csv', delimiter=',')
d_test = np.loadtxt('test.arff.csv', delimiter=',')
X = d[: , 0:92]
y = d[:, 92:93]
dummy = np.ones (X.shape[0])
dummy_1 = np.ones (d_test.shape[0])
Phi = np.column_stack([dummy, X])

Phi_test = np.column_stack([dummy_1, d_test])
#%%
from sklearn.linear_model import perceptron
from sklearn import svm

D = perceptron.Perceptron(n_iter=1000, verbose=0, random_state=9, fit_intercept=True, eta0=0.01)
D.fit(Phi,y)
y_train_pred=D.predict(Phi)
# Print the results
print ("Prediction " + str(D.predict(Phi)))
print ("Actual     " + str(y))

print ("Training Accuracy   " + str(D.score(Phi, y)*100) + "%")
print("Training error "+ str(np.sum(y!=y_train_pred.reshape(y.shape))/np.float(y_train_pred.shape[0])))
w=D.coef_
w=w.T
y_test_pred = D.predict(Phi_test)


np.savetxt('TrigramPrediction.csv', y_test_pred, fmt='%i', delimiter = ',')
#np.savetxt('TrigramPrediction.csv')

#%%
D_svm = svm.SVC()
D_svm.fit(Phi,y)
y_train_pred=D.predict(Phi)
# Print the results
print ("Prediction " + str(D.predict(Phi)))
print ("Actual     " + str(y))

print ("Training Accuracy   " + str(D.score(Phi, y)*100) + "%")
#print("Training error "+ str(np.sum(y!=y_train_pred)/np.float(y_train_pred.shape[0])))
w=D.coef_
w=w.T
y_test_pred = D.predict(Phi_test)

#%%
from sklearn.model_selection import train_test_split
  
X_train, X_heldout, y_train, y_heldout = train_test_split(X, y, test_size=0.35, random_state=0)
print(X_train.shape,X_heldout.shape, y_train.shape, y_heldout.shape)

x_dummy_train = np.ones(X_train.shape[0])
x_dummy_heldout = np.ones(X_heldout.shape[0])

Phi_train = np.column_stack((x_dummy_train,X_train))
Phi_heldout  = np.column_stack((x_dummy_heldout,X_heldout))

#%%
w_hat = np.zeros(Phi_train.shape[1])

T = 50
D = perceptron.Perceptron(n_iter=100, verbose=0, random_state=7, fit_intercept=True, eta0=0.05)

train_error = np.zeros(T)
heldout_error = np.zeros(T)
for ep in range(T):
    # here we use a learning rate, which decays with each epoch
    D.partial_fit(Phi_train, y_train, classes = np.unique(y_train))
    w_hat=D.coef_
    w_hat = w_hat.T
    y_train_pred = D.predict(Phi_train)
    y_test_pred = D.predict(Phi_heldout)
    train_error[ep] = np.sum(y_train_pred.reshape(y_train.shape) != y_train) / np.float(y_train.shape[0])
    heldout_error[ep] = np.sum(y_test_pred.reshape(y_heldout.shape) != y_heldout) / np.float(y_heldout.shape[0])
plt.figure()
plt.plot(train_error, label = 'Train Error')
plt.plot(heldout_error, label = 'Heldout Error')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.savefig('Exp1.eps', dpi =1000)
print("Average test error "+ str(heldout_error.mean()))
print("Average train error "+ str(train_error.mean()))
