from __future__ import division
from __future__ import print_function
import pandas as pd
import numpy as np
import random
from sklearn.utils import shuffle

dataset=pd.read_csv("C:/Users/giannis/Desktop/on-line ML/datasets/ES_correct.txt")
dataset.insert(0,'X1',1)
sum_errors=[0,0,0,0,0,0,0,0,0,0,0,0]

for i in range(10):
    dataset=shuffle(dataset)
    X=dataset.iloc[:,:-1]
    y=dataset.iloc[:,-1]
    
    
    #parameter tuning
    n=.05
    w=np.zeros(len(dataset.columns)-1)
    '''
    w=[]
    for i in range(len(X.columns)):
        w.append(random.uniform(-1,1))
    '''
    model_predictions=[1]*11500
        
    X=np.array(X)
    w=np.array(w)
    y=np.array(y)
    
    for i in range(0,len(dataset)):
        x=X[i,:]
        y_predicted=np.sign(np.dot(x,w))
        y_real=y[i]
        model_predictions[i]=y_predicted
        if ( y_predicted!=y_real ):  
            w=w+(n*y[i]*x)
           
    #find number of online errors
    errors=[]
    error_counter=0
    
    
    for i in range(len(dataset)-1):
        if y[i]!=model_predictions[i]: error_counter+=1
        if i%1000==0: errors.append(error_counter)
    
    errors.append(error_counter) #append final element
    errors=errors[1:]
    
    for i in range(len(errors)):
        sum_errors[i]+=errors[i]
        
for i in range(len(sum_errors)):
    sum_errors[i]=sum_errors[i]/10
            

percent=[]
a= sum_errors

percent.append(round(a[0]/1000, 4))
percent.append(round(a[1]/2000, 4))
percent.append(round(a[2]/3000, 4))
percent.append(round(a[3]/4000, 4))
percent.append(round(a[4]/5000, 4))
percent.append(round(a[5]/6000, 4))
percent.append(round(a[6]/7000, 4))
percent.append(round(a[7]/8000, 4))
percent.append(round(a[8]/9000, 4))
percent.append(round(a[9]/10000, 4))
percent.append(round(a[10]/11000, 4))
percent.append(round(a[11]/11500, 4))


'''
offline errors: [270, 522, 787, 1060, 1343, 1628, 1898, 2177, 2434, 2723, 2840]
'''

absolute_regret=[]
a=a[1:]
off=[270, 522, 787, 1060, 1343, 1628, 1898, 2177, 2434, 2723, 2840]

absolute_regret.append(round(abs(a[0]-off[0])/2000, 4))
absolute_regret.append(round(abs(a[1]-off[1])/3000, 4))
absolute_regret.append(round(abs(a[2]-off[2])/4000, 4))
absolute_regret.append(round(abs(a[3]-off[3])/5000, 4))
absolute_regret.append(round(abs(a[4]-off[4])/6000, 4))
absolute_regret.append(round(abs(a[5]-off[5])/7000, 4))
absolute_regret.append(round(abs(a[6]-off[6])/8000, 4))
absolute_regret.append(round(abs(a[7]-off[7])/9000, 4))
absolute_regret.append(round(abs(a[8]-off[8])/10000, 4))
absolute_regret.append(round(abs(a[9]-off[9])/11000, 4))
absolute_regret.append(round(abs(a[10]-off[10])/11500, 4))

print("regret: ", percent)
print( "absolute regret:", absolute_regret)









