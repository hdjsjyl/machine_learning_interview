# step1 load dataset
# step2 make dataset iterable
# step3 create model class
# step4 instantiate model class
# step5 instantiate cost function
# step6 instantiate optimizer
# step7 train model
# step8 test model


import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.utils import shuffle


# >> FEATURE SELECTION << #
def remove_correlated_features(X):
    corr_threshold = 0.9
    corr = X.corr()
    drop_columns = np.full(corr.shape[0], False, dtype=bool)
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[0]):
            if corr.iloc[i, j] >= corr_threshold:
                drop_columns[j] = True
    columns_dropped = X.columns[drop_columns]
    X.drop(columns_dropped, axis=1, inplace=True)
    return columns_dropped


def remove_less_significant_features(X, Y):
    sl = 0.05
    regression_ols = None
    columns_dropped = np.array([])
    for itr in range(0, len(X.columns)):
        regression_ols = sm.OLS(Y, X).fit()
        max_col = regression_ols.pvalues.idxmax()
        max_val = regression_ols.pvalues.max()
        if max_val > sl:
            X.drop(max_col, axis='columns', inplace=True)
            columns_dropped = np.append(columns_dropped, [max_col])
        else:
            break
    regression_ols.summary()
    return columns_dropped

def datapreprocessing():
    data = pd.read_csv('./data/data.csv')

    # drop last column (extra column added by pd)
    # and unnecessary first column (id)
    data.drop(data.columns[[-1, 0]], axis=1, inplace=True)

    print("applying feature engineering...")
    # convert categorical labels to numbers
    diag_map = {'M': 1.0, 'B': -1.0}
    data['diagnosis'] = data['diagnosis'].map(diag_map)

    # put features & outputs in different data frames
    Y = data.loc[:, 'diagnosis']
    X = data.iloc[:, 1:]

    # filter features
    remove_correlated_features(X)
    remove_less_significant_features(X, Y)

    # normalize data for better convergence and to prevent overflow
    X_normalized = MinMaxScaler().fit_transform(X.values)
    X = pd.DataFrame(X_normalized)

    # insert 1 in every row for intercept b
    X.insert(loc=len(X.columns), column='intercept', value=1)

    # split data into train and test set
    print("splitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = tts(X, Y, test_size=0.2, random_state=42)
    # print(X_train.shape, y_train.shape)
    return X_train, X_test, y_train, y_test


def hingeloss(output, y, lamd):
    ## hinge loss: max(0, 1-output)
    distance = 1 - y*output
    # print(distance[:20])
    distance[distance < 0] = 0

    hingeLoss = lamd*(np.sum(distance)/len(y))
    # print(hingeLoss)
    ## l2 regularization
    # regularization = (1/2)*np.dot(w.T, w)
    return hingeLoss, distance

def model(x, w, b):
    z = np.sum(x * w.T + b, axis=1)
    z = (z)[:, None]
    return z

def optimizer(x, y, w, b, lamb, mask):
    y = y*mask
    dw1 = lamb * np.sum(-y*x, axis=0)/len(y)
    dw1 = (dw1)[:, None]
    # dw2 = w
    db = lamb * np.sum(-y)/len(y) * mask
    return dw1, db

def dataLoader(x, y, batchsize):
    indices = [j for j in range(len(x_train))]
    np.random.shuffle(indices)
    for i in range(0, len(x), batchsize):
        x1 = x_train[indices[i:min(i+batchsize, len(x))]]
        y1 = y_train[indices[i:min(i+batchsize, len(x))]]
        yield x1, y1


if __name__ == '__main__':
    X_train, X_test, Y_train, Y_test = datapreprocessing()
    x_train = X_train.values
    y_train = Y_train.values

    num_features = x_train.shape[1]
    y_train = (y_train)[:, None]
    # w = np.random.randn(num_features, 1)
    w = np.zeros((num_features, 1))
    # print(w[:5])
    b = 0
    # print(x_train.shape)
    regularization_strength = 0.1
    lr = 0.00001
    iters = 5000
    for iter in range(iters):
        tmp = 0
        for x1, y1 in dataLoader(x_train, y_train, batchsize=91):
            y_hat = model(x1, w, b)
            # print(y_hat[:5])
            loss, mask = hingeloss(y_hat, y1, regularization_strength)
            # print(loss.shape)
            dw, db = optimizer(x1, y1, w, b, regularization_strength, mask)
            w -= lr*dw
            b -= lr%db
            tmp += loss
        print('iter: {}'.format(iter), 'loss: {}'.format(tmp/5))

