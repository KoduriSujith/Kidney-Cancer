import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

##dataset
data= pd.read_csv("C:\\Users\\Sujith\\Desktop\\pd_speech_features.csv")
##‪data=pd.read_csv("C:\\Users\\Sujith\\Desktop\\KIDNEY CANCER.csv")

data.columns = data.iloc[0]
data=data.iloc[1:,:]
print(data.head(5))
x = data.drop(columns=['class'])
y = data['class']
ob= SVC()
pca = PCA(10)
pca.fit(x)
x = pca.transform(x)
print(x.shape)
##splitting
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y)
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

##perceptron
class Perceptron(object):

        def __init__(self, eta, n_iter):
                self.eta = eta
                self.n_iter = n_iter

        def fit(self, X, y):
                self.w_ = np.zeros(1 + X.shape[1])
                self.errors_=[]
                for _ in range(self.n_iter):
                        errors=0
                        for xi, target in zip(X, y):
                                s=self.predict(xi)
                                error = int(target) - s
                                if error != 0:
                                        update = self.eta * error
                                        self.w_[1:] += update * xi
                                        self.w_[0] += update
                                        errors+=int(update!=0.0)
                        self.errors_.append(errors)
                return self

        def net_input(self, X):
                return np.dot(X, self.w_[1:]) + self.w_[0]

        def predict(self, X):
                return np.where(self.net_input(X) >= 0.0, 0, 1)

import matplotlib.pyplot as plt
model = Perceptron(n_iter=1000,eta=0.1)
model.fit(x_train, y_train)
c=0
y_pred = model.predict(x_test)
print('misclassified samples: %d'%(y_test!=y_pred).sum())#compute

##accuracy
for i,j in zip(y_test,y_pred):
    if int(i)==j:
        c+=1
print(c/len(y_test))
