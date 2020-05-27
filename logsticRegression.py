## logistic regression, 对数几率回归， 解决分类问题
#  但叫成回归的原因是来自于一般回归模型， y = g(ax+b)
# 这里的g是sigmoid函数，损失函数是通过几大似然估计得到的
# p(y|x,o) = (y_hat).power(yi)*(1-y_hat).power(1-yi)
# y_hat sigmoid预测值， yi为对应的ground truth分类
# 似然函数L
# L = TT(y_hat).power(yi)*(1-y_hat).power(1-yi), i=1...m
# logL = Eyi*log(y_hat)+(1-yi)*log(1-y_hat), i=1...m
# The gradient of sigmoid cross entropy / w = (y_hat-y)*x


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# %matplotlib inline
from sklearn.datasets import load_breast_cancer
#from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

dataset = load_breast_cancer()
data = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
data['cancer'] = [dataset.target_names[t] for t in dataset.target]

def sigmoid(z):
    s = 1/(1+np.exp(-z))
    s = s.reshape(s.shape[0],1)
    return s

def model(theta, b, X):
    z = np.sum(X*theta.T+b, axis=1)
    return sigmoid(z)

#cross_entropy
def cross_entropy(y, y_hat):
    n_samples = y.shape[0]
    # print(y_hat.shape, y.shape)
    return sum(-y*np.log(y_hat)-(1-y)*np.log(1-y_hat))/n_samples

def cost_function(theta, b, X, y):
    y_hat = model(theta, b, X)
    return cross_entropy(y, y_hat)

def optimize(theta,b,X,y):
    n = X.shape[0]
    alpha = 1e-1
    y_hat = model(theta,b,X)
    dtheta = (1.0/n) * ((y_hat-y)*X)
    # print(dtheta.shape)
    dtheta = np.sum(dtheta, axis=0)
    dtheta=dtheta.reshape((30,1))
    theta = theta - alpha * dtheta
    db     = (1.0/n) * (y_hat-y)
    db     = np.sum(db, axis=0)
    b      = b - alpha * db
    return theta, b

def predict_proba(theta, b, X):
    y_hat=model(theta, b, X)
    return y_hat

def predict(X, theta, b):
    y_hat=predict_proba(theta,b,X)
    y_hard=(y_hat > 0.5)
    return y_hard

def accuracy(theta, b, X, y):
    y_hard=predict(X, theta, b)
    count_right=sum(y_hard == y)
    print(count_right.shape)
    return count_right/len(y)

def accuracy2(theta, b, x, y):
    y_hat = model(theta, b, x)
    y_hard = (y_hat > 0.5)
    return sum(y_hard == y)/len(y)


def iterate(theta,b,X,y,times):
    costs = []
    accs = []
    for i in range(times):
        theta = optimize(theta,b,X,y)
        costs.append(cost_function(theta, b, X, y))
        accs.append(accuracy(theta, b, X, y))
        print(costs[-1], accs[-1])
    return theta, b, costs, accs


X = dataset.data
y = dataset.target
n_features = X.shape[1]

std=X.std(axis=0)
mean=X.mean(axis=0)
X_norm = (X - mean) / std


def add_ones(X):
    ones=np.ones((X.shape[0],1))
    X_with_ones=np.hstack((ones, X))
    return X_with_ones

# X_with_ones = add_ones(X_norm)

X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size = 0.3, random_state=12345)
y_train=y_train.reshape((y_train.shape[0],1))
y_test=y_test.reshape((y_test.shape[0],1))

theta = np.ones((n_features,1))
b = 0

# theta, b, costs, accs = iterate(theta, b, X_train, y_train, 1500)
alpha = 1e-1
costs = []
accs = []
for iter in range(1500):
    # theta_new,b_new = optimize(theta, b, X_train, y_train)
    # theta = theta_new
    # b = b_new
    n = X.shape[0]
    y_hat = model(theta, b, X_train)
    ## update gradients
    dtheta = (1.0 / n) * ((y_hat - y_train) * X_train)
    # print(dtheta.shape)
    dtheta = np.sum(dtheta, axis=0)
    dtheta = dtheta.reshape((30, 1))
    theta = theta - alpha * dtheta
    db = (1.0 / n) * (y_hat - y_train)
    db = np.sum(db, axis=0)
    b = b - alpha * db
    # costs.append(cost_function(theta, b, X_train, y_train))
    # y_hat = model(theta, b, X_train)
    loss = cross_entropy(y_train, y_hat)
    acc = accuracy(theta, b, X_train, y_train)
    print(loss, acc)

# print(costs[-1], accs[-1])

accuracy(theta, b, X_test, y_test)