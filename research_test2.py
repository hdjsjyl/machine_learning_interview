## 1.
import pandas as pd
import numpy as np
import time# visual libraries
from matplotlib import pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('ggplot')# sklearn libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,matthews_corrcoef,classification_report,roc_curve
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

## 2.
df = pd.read_csv('../input/creditcard.csv')
## return first 5 rows data
df.head()

## 3.
## check the number of null values
df.isnull().any().sum()
## check number of rows
All = df.shape[0]
## get data with label of fraud
fraud = df[df['Class'] == 1]
## get dta with label no fraud
nonFraud = df[df['Class'] == 0]

## check percentage of fraud and nofraud data
x = len(fraud) / All
y = len(nonFraud) / All

print('frauds :', x * 100, '%')
print('non frauds :', y * 100, '%')

labels = ['non frauds', 'fraud']
## count how many different values in the table data and get the number of each value
classes = pd.value_counts(df['Class'], sort=True)
classes.plot(kind='bar', rot=0)
plt.title("Transaction class distribution")
plt.xticks(range(2), labels)
plt.xlabel("Class")
plt.ylabel("Frequency")

## 4.

# distribution of Amount
amount = [df['Amount'].values]
sns.distplot(amount)

time = df['Time'].values
sns.distplot(time)

anomalous_features = df.iloc[:, 1:29].columns

plt.figure(figsize=(12, 28 * 4))
gs = gridspec.GridSpec(28, 1)
for i, cn in enumerate(df[anomalous_features]):
    ax = plt.subplot(gs[i])
sns.distplot(df[cn][df.Class == 1], bins=50)
sns.distplot(df[cn][df.Class == 0], bins=50)
ax.set_xlabel('')
ax.set_title('histogram of feature: ' + str(cn))
plt.show()
correlation_matrix = df.corr()
fig = plt.figure(figsize=(12, 9))
sns.heatmap(correlation_matrix, vmax=0.8, square=True)
plt.show()


## 5.
df['Vamount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))
df['Vtime'] = StandardScaler().fit_transform(df['Time'].values.reshape(-1, 1))

df = df.drop(['Time', 'Amount'], axis=1)
df.head()

## 6.
X = df.drop(['Class'], axis=1)
y = df['Class']

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X.values)
principalDf = pd.DataFrame(data=principalComponents
                           , columns=['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, y], axis=1)
finalDf.head()

# 2D visualization
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Principal Component 1', fontsize=15)
ax.set_ylabel('Principal Component 2', fontsize=15)
ax.set_title('2 component PCA', fontsize=20)
targets = [0, 1]
colors = ['r', 'g']
for target, color in zip(targets, colors):
    indicesToKeep = finalDf['Class'] == target
ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
           , finalDf.loc[indicesToKeep, 'principal component 2']
           , c=color
           , s=50)
ax.legend(targets)
ax.grid()

df = df.sample(frac=1)

frauds = df[df['Class'] == 1]
non_frauds = df[df['Class'] == 0][:492]

new_df = pd.concat([non_frauds, frauds])
# Shuffle dataframe rows
new_df = new_df.sample(frac=1, random_state=42)
# Let's plot the Transaction class against the Frequency
labels = ['non frauds', 'fraud']
classes = pd.value_counts(new_df['Class'], sort=True)
classes.plot(kind='bar', rot=0)
plt.title("Transaction class distribution")
plt.xticks(range(2), labels)
plt.xlabel("Class")
plt.ylabel("Frequency")

# prepare the data
features = new_df.drop(['Class'], axis=1)
labels = pd.DataFrame(new_df['Class'])

feature_array = features.values
label_array = labels.values

## 7.
X_train, X_test, y_train, y_test = train_test_split(feature_array, label_array, test_size=0.20)

X_train = normalize(X_train)
X_test = normalize(X_test)

neighbours = np.arange(1, 25)
train_accuracy = np.empty(len(neighbours))
test_accuracy = np.empty(len(neighbours))

for i, k in enumerate(neighbours):
    knn = KNeighborsClassifier(n_neighbors=k, algorithm="kd_tree", n_jobs=-1)

knn.fit(X_train, y_train.ravel())

train_accuracy[i] = knn.score(X_train, y_train.ravel())

test_accuracy[i] = knn.score(X_test, y_test.ravel())

plt.title('k-NN Varying number of neighbors')
plt.plot(neighbours, test_accuracy, label='Testing Accuracy')
plt.plot(neighbours, train_accuracy, label='Training accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()

idx = np.where(test_accuracy == max(test_accuracy))
x = neighbours[idx]

knn = KNeighborsClassifier(n_neighbors=x[0], algorithm="kd_tree", n_jobs=-1)
knn.fit(X_train, y_train.ravel())

filename = 'finalized_model.sav'
joblib.dump(knn, filename)  # load the model from disk
knn = joblib.load(filename)

knn_predicted_test_labels = knn.predict(X_test)

from pylab import rcParams

rcParams['figure.figsize'] = 14, 8
plt.subplot(222)
plt.scatter(X_test[:, 0], X_test[:, 1], c=knn_predicted_test_labels)
plt.title(" Number of Blobs")

knn_accuracy_score = accuracy_score(y_test, knn_predicted_test_labels)
knn_precison_score = precision_score(y_test, knn_predicted_test_labels)
knn_recall_score = recall_score(y_test, knn_predicted_test_labels)
knn_f1_score = f1_score(y_test, knn_predicted_test_labels)
knn_MCC = matthews_corrcoef(y_test, knn_predicted_test_labels)
