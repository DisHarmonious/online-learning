import pandas as pd
import numpy as np
from sklearn import preprocessing

#import data
dataset=pd.read_csv("C:/Users/giannis/Desktop/on-line ML/datasets/data.csv")
dataset=dataset.iloc[:,1:]
X=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]

max_abs_scaler = preprocessing.MaxAbsScaler()
X = max_abs_scaler.fit_transform(X)


for i in range(len(dataset)):
    if y[i]!=1: y[i]=-1


new=np.column_stack((X,y))
np.savetxt("C:/Users/giannis/Desktop/ES_correct.txt", new, delimiter=',')

