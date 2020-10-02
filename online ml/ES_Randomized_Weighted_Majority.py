import pandas as pd
import numpy as np
import random

dataset=pd.read_csv("C:/Users/giannis/Desktop/on-line ML/datasets/ES_correct.txt")
X=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]

############################### TRAIN THE EXPERTS#########################
##########################################################################

########################### EXPERT 1: Perceptron
n=0.05
w=np.zeros(len(X.columns))

def weight_update(w,x,n,y):
	new_weight=w+sum(np.dot(n*y,x))
	return new_weight


for i in range(0,7000):
    x=X.iloc[i,:]
    y_predicted=np.sign(np.dot(x,w))
    y_real=y[i]
    if ( y_predicted!=y_real ):  
        for j in range(0,len(dataset.columns)-1):
            w[j]=weight_update(w[j],x,n,y_real)
            
model1_weights=w

###################### EXPERT 2: SVM
C=1000000
n=0.05
w=[]
for i in range(len(X.columns)):
	w.append(random.uniform(-1,1))


def weight_update_with_C(w,x,n,y,c):
	new_weight = w - n*w + sum(np.dot(n*c*y,x))
	return new_weight
def weight_update(w,n):
	new_weight=w-n*w
	return new_weight

for i in range(0,7000):
    	x=X.iloc[i,:]
    	y_predicted=np.sign(np.dot(x,w))
        y_real=y[i]
        check=y_real*np.dot(w,x)
    	if ( check<1 ):
        		for j in range(len(dataset.columns)-1):
        			w[j]=weight_update_with_C(w[j],x,n,C,y_real)
    	else: 
    		for j in range(0,len(dataset.columns)-1):
    			w[j]=weight_update(w[j],n)
            
model2_weights=w

############### EXPERT 3: Logistic Regression
n=0.05
threshold=.5
w=[]
for i in range(len(X.columns)):
    w.append(random.uniform(-1,1))


for i in range(0,7000):
    	hypothesis=1/(1+np.exp(np.dot(w,X.iloc[i,:])))
    	if hypothesis>threshold: y_predicted=1
    	else: y_predicted=-1
    	if y_predicted!=y[i]:
    		for j in range(len(dataset.columns)-1):
    			w[j]-=n*(hypothesis-y.iloc[i])*X.iloc[i,j]

model3_weights=w
threshold1=threshold
            
################# EXPERT 4: ANOTHER PERCEPTRON
n=0.05
w=np.zeros(len(X.columns))

def weight_update(w,x,n,y):
	new_weight=w+sum(np.dot(n*y,x))
	return new_weight


for i in range(0,7000):
    x=X.iloc[i,:]
    y_predicted=np.sign(np.dot(x,w))
    y_real=y[i]
    if ( y_predicted!=y_real ):  
        for j in range(0,len(dataset.columns)-1):
            w[j]=weight_update(w[j],x,n,y_real)
            
model4_weights=w 
           
###################### EXPERT 5: ANOTHER LOGISTIC REGRESSION
n=0.05
threshold=.4
w=[]
for i in range(len(X.columns)):
    w.append(random.uniform(-1,1))
errors=0
e=[]
   	   
for i in range(0,7000):
    	hypothesis=1/(1+np.exp(np.dot(w,X.iloc[i,:])))
    	if hypothesis>threshold: y_predicted=1
    	else: y_predicted=-1
    	if y_predicted!=y[i]:
    		for j in range(len(dataset.columns)-1):
    			w[j]-=n*(hypothesis-y.iloc[i])*X.iloc[i,j]

model5_weights=w  
threshold2=threshold          
            
            
######################################### START PREDICTING WITH THE EXPERT###################################
#############################################################################################################
expert_weights=np.ones(5)
prediction=np.zeros(5)
probability=[.2,.2,.2,.2,.2]
beta=0.75
errors=0
e=[]

for i in range(7001,len(dataset)):
    x_new=X.iloc[i]
    ######### CREATE EXPERT PREDICTIONS
    prediction[0]=np.sign(np.dot(x_new,model1_weights))
    prediction[1]=np.sign(np.dot(x_new,model2_weights))
    prediction[3]=np.sign(np.dot(x_new,model4_weights))
    hypothesis=1/(1+np.exp(np.dot(model3_weights,x_new)))
    if hypothesis>threshold1: prediction[2]=1
    else: prediction[2]=-1
    hypothesis=1/(1+np.exp(np.dot(model5_weights,x_new)))
    if hypothesis>threshold2: prediction[4]=1
    else: prediction[4]=-1 
    ######### MAKE THE FINAL PREDICTION
    y_predicted=np.random.choice(prediction, 1, probability)
    ######### ADJUST NEW EXPERT WEIGHTS
    for j in range(len(expert_weights)):
        if prediction[j]!=y[i]: expert_weights[j]=expert_weights[j]*beta
    weight_sum=sum(expert_weights)
    for j in range(len(expert_weights)):
        probability[j]=expert_weights[j]/weight_sum
    ######## calculate accuracy
    if y_predicted!=y[i]: errors+=1
    if i%100==0:
        e.append(errors)
        errors=0

