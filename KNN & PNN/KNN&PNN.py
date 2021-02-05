# -*- coding: utf-8 -*-

# DO NOT CHANGE
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances
import time
import matplotlib.pyplot as plt

#%%
def wkNN(Xtr,ytr,Xts,k,random_state=None):
    # Implement weighted kNN
    # Xtr: training input data set
    # ytr: ouput labels of training set
    # Xts: test data set
    # random_state: random seed
    # return 1-D array containing output labels of test set
       
    d = pairwise_distances(Xts,Xtr,metric='euclidean')
    
    y_pred=[]
    for i in range(0,len(d)):
        nn=np.argsort(d[i])[0:k]
     
        w={}
        w[0]=0
        w[1]=0
        w[2]=0
        for j in range(0,len(nn)):
            if d[i,nn[-1]] != d[i,nn[0]] :
                w[ytr[nn[j]]]=w[ytr[nn[j]]]+((d[i,nn[-1]]-d[i,nn[j]])/(d[i,nn[-1]]-d[i,nn[0]]))
             
                
            else :
               w[ytr[nn[j]]]=1
            
        
        y_pred.append(max(w.keys(), key=lambda a : w[a]))
            
    return y_pred

#%%

def PNN(Xtr,ytr,Xts,k,random_state=None):
    # Implement PNN
    # Xtr: training input data set
    # ytr: ouput labels of training set
    # Xts: test data set
    # random_state: random seed
    # return 1-D array containing output labels of test set
    
    d= pairwise_distances(Xts,Xtr,metric='euclidean')
    y_pred=[]   
    y_class=np.unique(ytr)
    
    
    for j in range(0,len(d)):
        nn_2={}
        for i in y_class:
            nn=d[j,np.where(ytr==i)]
            nn_2[i]=list(nn[0][np.argsort(nn[0])[0:k]])
    
        for s in range(0,len(nn_2)):
            nn_3=[]
            for n in range(0,len(nn_2[s])):
                nn_3.append(nn_2[s][n]*(1/(n+1)))
            nn_2[s]=sum(nn_3)
    
    
        y_pred.append(min(nn_2.keys(), key=lambda a : nn_2[a]))
    
    
       
    return y_pred

   
#%%

X1,y1=datasets.make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_classes=3, n_clusters_per_class=1, random_state=13)
Xtr1,Xts1, ytr1, yts1=train_test_split(X1,y1,test_size=0.2, random_state=22)

#TODO: Cacluate accuracy with varying k for wkNN and PNN
#TODO: Calculate computation time
#TODO: Draw scatter plot

kk=[3,5,7,9,11]
wknn=[]
for i in range(0,len(kk)) :
    start1 = time.time()
    pred = wkNN(Xtr1,ytr1,Xts1,kk[i])
    acc=[]
    
    for j in range(0,200):
        if pred[j]==yts1[j] :
            acc.append(1)
            
        else :
            acc.append(0)
            
    wknn.append(acc.count(1)/len(acc))
end1=time.time()
compute1 = end1-start1    

pnn=[]
for ii in range(0,len(kk)) :
    start2=time.time()
    pred1 = PNN(Xtr1,ytr1,Xts1,kk[ii])
    acc1=[]
    
    for jj in range(0,200):
        if pred1[jj]==yts1[jj] :
            acc1.append(1)
            
        else :
            acc1.append(0)
            
    pnn.append(acc1.count(1)/len(acc1))
end2 = time.time()
compute2= end2-start2

#%%

X2,y2=datasets.make_classification(n_samples=1000, n_features=6, n_informative=2, n_redundant=3, n_classes=2, n_clusters_per_class=2, flip_y=0.2,random_state=75)
Xtr2,Xts2, ytr2, yts2=train_test_split(X2,y2,test_size=0.2, random_state=78)

#TODO: Cacluate accuracy with varying k for wkNN and PNN
#TODO: Calculate computation time
#TODO: Draw scatter plot

wknn2=[]
for i in range(0,len(kk)) :
    start11 = time.time()
    pred = wkNN(Xtr2,ytr2,Xts2,kk[i])
    acc=[]
    
    for j in range(0,200):
        if pred[j]==yts2[j] :
            acc.append(1)
            
        else :
            acc.append(0)
            
    wknn2.append(acc.count(1)/len(acc))
end11=time.time()
compute11 = end11-start11    

pnn2=[]
for ii in range(0,len(kk)) :
    start22=time.time()
    pred1 = PNN(Xtr2,ytr2,Xts2,kk[ii])
    acc1=[]
    
    for jj in range(0,200):
        if pred1[jj]==yts2[jj] :
            acc1.append(1)
            
        else :
            acc1.append(0)
            
    pnn2.append(acc1.count(1)/len(acc1))
end22 = time.time()
compute22= end22-start22

#%%

# k에 해당하는 wkNN, PNN acc 출력

print('Elapsed time: {:.4f}'.format(compute1 + compute2))
print('------'*5)
print('{:>5}{:^15}{:^6}'.format('k', 'wkNN', 'PNN'))
print('------'*5)
for t in range(len(kk)):
    print('{:>5}{:^15.4f}{:^6.4f}'.format(kk[t], wknn[t], pnn[t]))
print('------'*5)

print('Elapsed time: {:.4f}'.format(compute11 + compute22))
print('------'*5)
print('{:>5}{:^15}{:^6}'.format('k', 'wkNN', 'PNN'))
print('------'*5)
for t in range(len(kk)):
    print('{:>5}{:^15.4f}{:^6.4f}'.format(kk[t], wknn2[t], pnn2[t]))
print('------'*5)

#%%

# scatter plot 1 정의

kkk=[7]
for i in range(0,len(kkk)) :
    start1 = time.time()
    pred = wkNN(Xtr1,ytr1,Xts1,kk[i])
    acc_scatter=[]
    
    for j in range(0,200):
        if pred[j]==yts1[j] :
            acc_scatter.append(1)
            
        else :
            acc_scatter.append(0)
            
acc_scatter=np.array(acc_scatter)   
miss1=Xts1[np.where(acc_scatter==0),:]

for ii in range(0,len(kkk)) :
    start2=time.time()
    pred1 = PNN(Xtr1,ytr1,Xts1,kk[ii])
    acc1_scatter=[]
    
    for jj in range(0,200):
        if pred1[jj]==yts1[jj] :
            acc1_scatter.append(1)
            
        else :
            acc1_scatter.append(0)

acc1_scatter=np.array(acc1_scatter)   
miss2=Xts1[np.where(acc1_scatter==0),:]

#%%

#scatter plot 2 정의

for i in range(0,len(kkk)) :
    start11 = time.time()
    pred = wkNN(Xtr2,ytr2,Xts2,kk[i])
    acc_scatter1=[]
    
    for j in range(0,200):
        if pred[j]==yts2[j] :
            acc_scatter1.append(1)
            
        else :
            acc_scatter1.append(0)

acc_scatter1=np.array(acc_scatter1)   
miss1=Xts1[np.where(acc_scatter1==0),:]           


for ii in range(0,len(kkk)) :
    start22=time.time()
    pred1 = PNN(Xtr2,ytr2,Xts2,kk[ii])
    acc1_scatter2=[]
    
    for jj in range(0,200):
        if pred1[jj]==yts2[jj] :
            acc1_scatter2.append(1)
            
        else :
            acc1_scatter2.append(0)
acc1_scatter2=np.array(acc1_scatter2)   
miss22=Xts1[np.where(acc1_scatter2==0),:]

#%%

# scatter plot 1,2 출력

plt.figure(figsize=(10, 8))
plt.scatter(Xtr1[:,0],Xtr1[:,1],c=ytr1,label='Train')
plt.scatter(Xts1[:,0],Xts1[:,1],c=yts1,marker='x',label='Test')
plt.scatter(miss1[0][:,0],miss1[0][:,1],marker='s',edgecolors='r',facecolors='none',label='Missclassified by wkNN')
plt.scatter(miss2[0][:,0],miss2[0][:,1],marker='d',edgecolors='b',facecolors='none',label='Missclassified by PNN')
plt.legend(loc='under right')


plt.figure(figsize=(10, 8))
plt.scatter(Xtr2[:,0],Xtr2[:,1],c=ytr2,label='Train')
plt.scatter(Xts2[:,0],Xts2[:,1],c=yts2,marker='x',label='Test')
plt.scatter(miss1[0][:,0],miss1[0][:,1],marker='s',edgecolors='r',facecolors='none',label='Missclassified by wkNN')
plt.scatter(miss22[0][:,0],miss22[0][:,1],marker='d',edgecolors='b',facecolors='none',label='Missclassified by PNN')
plt.legend(loc='upper right')

#%%