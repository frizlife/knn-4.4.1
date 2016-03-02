import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import random
from sklearn.neighbors import KNeighborsClassifier
import math
import numpy as np


data = pd.read_csv('iris.data', names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])

plt.suptitle('Sepal Length x Width', fontsize=18)
plt.xlabel('Sepal Length (cm)', fontsize=16)
plt.ylabel('Sepal Width (cm)', fontsize=16)
plt.scatter(data['sepal_length'], data['sepal_width'], alpha=0.5)
#plt.show()

#random point
xx = random.uniform(4, 9)
yy = random.uniform(2, 5)
point = [(xx, yy)]
print point

y = data['class'] #class 
X = data[['sepal_length', 'sepal_width']] #design matrix (normally)
#X = data[collist] = data[['sepal length', 'sepal width']] #design matrix (normally)

clf = KNeighborsClassifier(n_neighbors=10, weights = 'distance')
clf.fit(X, y)
preds = clf.predict(point)
print preds
plt.show()
