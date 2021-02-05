import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.metrics import confusion_matrix
import os
from collections import Counter
os.getcwd()
os.chdir('C:/Users/inolab/Desktop/심화기계학습')
import warnings
warnings.filterwarnings(action='ignore')


#%%
def local_k_acc(X_train,X_train_k):
    col=list(X_train.columns)
    col2=list(X_train.columns)
    col2.append('dist')
    for i in range(2,int(len(X_train)*0.1)+1):
        k=i
        k2=[]
        for ii in range(len(X_train)):
            point=list(X_train.iloc[ii,0:len(col)])
            X_train['dist'] = pairwise_distances([point],X_train,metric='euclidean')[0]
            sort_train = X_train.sort_values(by=['dist'],axis=0)
            kk=sort_train.iloc[range(k+1)]
            kk=kk.iloc[1:,:]
            kk_inx=list(kk.index)
            
            train_class=[]
            X_train=X_train.iloc[:,0:len(col)]
            
            for j in (kk_inx):
                train_class.append(Y_train[j])
                
            cnt=Counter(train_class)
            mode=cnt.most_common(1)
            mode_val=list(dict(cnt).values())
            mode1=mode[0][0]
            mode_ok=cnt[mode1]
            k2.append(round(int(mode_ok)/int(sum(mode_val)),2))
        
        
        X_train_k[str(k)+'k']=k2
    #X_train_k.drop(['sepal length','sepal width','petal length','petal width','dist'], axis=1, inplace=True)  
    X_train_k.drop(col2, axis=1, inplace=True)
    return(X_train_k,X_train)

#%%

def eval_acc(X_train):
    global_train=[]
    
    for jj in range(2,int(len(X_train)*0.1)+1):
        clf = knn(n_neighbors=jj)
        clf.fit(X_train, Y_train)
        global_train.append(round(clf.score(X_train,Y_train),2))
        
    for z in range(0,len(X_train_k)):
        X_train_k.iloc[z,:]=X_train_k.iloc[z,:]+global_train
    
    return(X_train_k)

#%%

def sample_k(X_train_k):
    sample_k=[]
    for zz in range(0,len(X_train_k)):
        inx=list(X_train_k.columns) 
        a=list(X_train_k.iloc[zz,:])
        aa=a.index(max(a))
        sample_k.append(inx[aa][0:-1])
    
    X_train_k['sample_k']=sample_k
    
    return(X_train_k)

#%%

def test_k(X_test, X_train, X_train_k):
    col=list(X_train.columns)
    
    k3=[]
    for k in range(0,len(X_test)):
        point=list(X_test.iloc[k,0:len(col)])
        X_train['dist'] = pairwise_distances([point],X_train,metric='euclidean')[0] 
        sort_train = X_train.sort_values(by=['dist'],axis=0)
        kk=sort_train.iloc[range(3)]
        kk_inx=list(kk.index)
        
        test_k=[]
        X_train=X_train.iloc[:,0:len(col)]
        for j in (kk_inx):
            test_k.append(X_train_k.loc[j]['sample_k'])
            
        cnt=Counter(test_k)
        mode=cnt.most_common(1)
        mode1=mode[0][0]
        k3.append(mode1)
        
    X_test['test_k']=k3
    return(X_test,X_train,X_train_k)

#%%
def fin_class(X_test, X_train ):
    col=list(X_train.columns)
    fin_class=[]
    for k in range(0,len(X_test)):
        point=list(X_test.iloc[k,0:len(col)])
        X_train['dist'] = pairwise_distances([point],X_train,metric='euclidean')[0] 
        sort_train = X_train.sort_values(by=['dist'],axis=0)
        kk=sort_train.iloc[range(int(X_test['test_k'].iloc[k]))]
        kk_inx=list(kk.index)
        
        local_class=[]
        X_train=X_train.iloc[:,0:len(col)]
        
        for j in (kk_inx):
            local_class.append(Y_train.loc[j])
        
        cnt=Counter(local_class)
        mode=cnt.most_common(1)
        mode1=mode[0][0]
        fin_class.append(mode1)    
            
    X_test['fin_class']=fin_class
    return(X_test, X_train)


#%%

data = pd.read_excel("balanced.xlsx") 
balanced = data.dropna()

X=balanced.iloc[:,0:-1]
Y=balanced['target']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state=1)

X_train_k=X_train


X_train_k, X_train = local_k_acc(X_train,X_train_k)

X_train_k=eval_acc(X_train)

X_train_k=sample_k(X_train_k)

X_test,X_train,X_train_k = test_k(X_test, X_train, X_train_k)

X_test, X_train = fin_class(X_test, X_train)

confusion=pd.DataFrame(data={'X_test':X_test['fin_class'],'Y_test':Y_test}).T

# 1행1열 : TP / 1행2열 : FN / 2행1열 : FP / 2행2열 : TN

# knn과 비교

knn_clf = knn(n_neighbors=10)
knn_clf.fit(X_train, Y_train)

confusion_matrix(knn_clf.predict(X_test), Y_test)

#%%

data1 = pd.read_excel("imbalanced.xlsx") 
imbalanced = data1.dropna()

X=imbalanced.iloc[:,0:-1]
Y=imbalanced['target']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state=1)

X_train_k=X_train


X_train_k, X_train = local_k_acc(X_train,X_train_k)

X_train_k=eval_acc(X_train)

X_train_k=sample_k(X_train_k)

X_test,X_train,X_train_k = test_k(X_test, X_train, X_train_k)

X_test, X_train = fin_class(X_test, X_train)

confusion=pd.DataFrame(data={'X_test':X_test['fin_class'],'Y_test':Y_test}).T

# 1행1열 : TP / 1행2열 : FN / 2행1열 : FP / 2행2열 : TN


# knn과 비교

knn_clf2 = knn(n_neighbors=10)
knn_clf2.fit(X_train, Y_train)

confusion_matrix(knn_clf2.predict(X_test), Y_test)


# imbalance data에서 일반 knn 보다 강세를 보임

#%%

