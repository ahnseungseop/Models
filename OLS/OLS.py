# -*- coding: utf-8 -*-

# Use only following packages
import numpy as np
from scipy import stats
from sklearn.datasets import load_boston
#%%
data=load_boston()
X=data.data
y=data.target

#%%
def ftest(X,y):
    # n,p 정의
    n=len(X)
    p=len(X[0])
    
    # 상수항 결합
    ints = np.ones(n).reshape(-1,1)
    X_data = np.concatenate((ints,X),axis=1)
    # 잔차의 제곱합이 최소가 되는 지점 구하기의 가중치 벡터 구하기 (잔차 제곱합 식을 w로 미분=0)
    X_T_X = np.matmul(X_data.T, X_data)

    weight = np.matmul(np.matmul(np.linalg.inv(X_T_X),X_data.T), y.reshape(-1,1))
    y_hat = np.matmul(X_data, weight)
    
    # 각각의 값 구하기
    SSE = sum((y.reshape(-1, 1) - y_hat)**2)

    SSR = sum((y_hat - np.mean(y_hat))**2)

    SST = (SSR + SSE)

    MSR = (SSR / p)

    MSE = (SSE / (n - p - 1))

    f_value = (MSR / MSE)

    p_value = (1- stats.f.cdf(f_value, p, n - p - 1))
    print("---------------------------------------------------------------------------------------")
    print("Factor         SS            DF           MS          F-value        Pr>F")
    print('Model     {a:.4f}         {b:.0f}        {c:.4f}     {d:.4f}       {e:.4f}'.format(a=SSR[0], b=p, c=MSR[0], d=f_value[0],e=p_value[0]))
    print("Error     {f:.4f}         {g:.0f}       {h:.4f}".format(f=SSE[0], g=n-p-1,h=MSE[0]))
    print("---------------------------------------------------------------------------------------")
    print("Total     {i:.4f}         {j:.0f}".format(i=SST[0], j= n-1))
    print("---------------------------------------------------------------------------------------")
    return 0

#%%
def ttest(X,y,varname=None):
    name =  np.append('constant', data.feature_names)
    n=len(X)
    p=len(X[0])
    
    # 상수항 결합
    ints = np.ones(n).reshape(-1,1)
    ints
    X_data = np.concatenate((ints,X),axis=1)
    
    # 잔차의 제곱합이 최소가 되는 지점 구하기의 가중치 벡터 구하기 (잔차 제곱합 식을 w로 미분=0)
    X_T_X = np.matmul(X_data.T, X_data)
    
    weight = np.matmul(np.matmul(np.linalg.inv(X_T_X),X_data.T), y.reshape(-1,1))
    y_hat = np.matmul(X_data, weight)
    
    SSE = sum((y.reshape(-1, 1) - y_hat)**2)
    
    SSR = sum((y_hat - np.mean(y_hat))**2)
    
    SST = SSR + SSE
    
    MSR = SSR / p
    
    MSE = SSE / (n - p - 1)
    
    se_beta = np.diag(MSE*(np.linalg.inv(np.matmul(X_data.T,X_data))))
    
    se=np.sqrt(se_beta)
    
    t_val = []
    for i in range(len(se_beta)):
        t_val.append(weight[i]/np.sqrt(se_beta[i]))
    
    p_val = ((1-stats.t.cdf(np.abs(np.array(t_val)),n-p-1))).reshape(-1,)
    
    print("---------------------------------------------------------------------------------------")
    print("   Variable      coef                 se                  t                      Pr>|t|")
    for i in range(0,14):
        print( "{e:^15} {a:<15.4f}     {b:<8.4f}          {c:<8.4f}                  {d:<8.4f}"
              .format(e=name[i],a=weight[i][0],b=se[i],c=t_val[i][0],d=(p_val[i]*2)))
        
    print("---------------------------------------------------------------------------------------")
    
    return 0
#%%
## Do not change!
# load data
data=load_boston()
X=data.data
y=data.target

ftest(X,y)
ttest(X,y,varname=data.feature_names)
