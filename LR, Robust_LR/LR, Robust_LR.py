#%%

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

#%%

# 데이터 다운로드
# 미국 주별 target : 석유소비량, feature : 석유세, 평균수입, 지역고속도로 길이, 전체인구중 운전면허자 비율
petrol = pd.read_excel('file location/petrol.xlsx')

X=petrol[['tax', 'income','highway','license']] # 설명변수는 2차원으로 입력
y=petrol['consumption']

lr = LinearRegression()
lr.fit(X, y)

print(lr.coef_) # 설명변수 각각의 추정된 회귀계수
print(lr.intercept_) # 절편

y_pred=lr.predict(X) # 예측값 출력

error=y-y_pred # 오차계산

print(lr.score(X,y)) # r-suqare 값 출력

#%%

# 회귀 문제에서 feature의 수가 많으면, 오히려 모델의 성능이 떨어지게 된다.
# 때문에 오차 추정시 penalty를 주어서 회귀계수 추정의 varience를 낮춰서 예측성능을 높이는 방법
# 또는, 분석에 용이하지 않은 회귀계수를 아예 없애버리는 방법이 있다.

# Robust Regression : Huber Regression
# Huber regression은 오차가 적을때는 목적함수에 기여하는 바가 제곱으로 증가하지만,
# 오차항이 클때는 선형적으로 증가하게 하는 회귀함수

hr=HuberRegressor(alpha=0, epsilon=1, max_iter=500) 
# epsilon : 오차가 제곱으로 증가하는 구간과 선형으로 증가하는 구간이 바뀌는 지점
# alpha : Regularization parameter (추정된 회귀계수에 대한 panalty) 
# max_iter : 회귀계수를 조금씩 바꾸어가며, 오차를 조정하는  최대 반복횟수

hr.fit(X,y)
hr.coef_
lr.coef_

# LR과 추정된 회귀계수의 차이가 있음 

hr.intercept_

hr.outliers_ # M이 너무 작으면 전부 True로 나옴, epsilon을 키우면 False도 출력됨
hr.scale_ # 오차항 

#%%

# Robust Regression : Ridge Regression
# LR의 목적함수에 각 회귀계수 제곱의 합 * 람다를 더해주는 회귀함수
# 람다가 커질수록 회귀계수의 추정치가 0에 근접하게 되어 회귀계수의 Varience가 낮아지게 됨
ridge=Ridge(alpha=1)
# alpha : 람다
ridge.fit(X,y)

ridge.coef_
ridge.intercept_


ridge2=Ridge(alpha=10)
ridge2.fit(X,y)

ridge2.coef_
ridge2.intercept_

# alpha를 키울수록 회귀계수가 감소함을 볼 수 있음.

#%%

# Robust Regression : Lasso Regression
# LR의 목적함수에 각 회귀계수 절대값의 합 * 람다를 더해주는 회귀함수
# Ridge와 다르게 회귀계수의 추정치가 0이 나오게 되어 모델을 단순하게 만들어 예측력을 높임

lasso=Lasso(alpha=1)
# alpha : 람다
lasso.fit(X,y)

lasso.coef_ 

lasso2=Lasso(alpha=10)
lasso2.fit(X,y)

lasso2.coef_
lasso2.intercept_

# 마지막 0으로 나온 값은 제거된 값이다. 
# Lasso는 Ridge와 다르게 penalty term을 늘리면, 회귀계수가 0에 수렴하게 된다.

#%%

# Outlier data에 대한 Huber와 Ridge의 반응 확인
# 시뮬레이션 데이터 생성
X,y=datasets.make_regression(n_samples=20, n_features=1, random_state=0, noise=4.0, bias=100.0)

plt.scatter(X.flatten(), y)


# 아웃라이어 값 추가
rng=np.random.RandomState(0)

X_outliers=rng.normal(0, 0.5,size=(4,1))

y_outliers=rng.normal(0,2.0,size=4)

X_outliers[:2,:]+=X.max()+X.mean()/4
X_outliers[2:,:]+=X.min()-X.mean()/4

y_outliers[:2]+=y.max()-y.mean()/4
y_outliers[2:]+=y.min()+y.mean()/4

X=np.vstack((X,X_outliers))
y=np.concatenate((y,y_outliers))

plt.scatter(X.flatten(),y)

#%%
# Huber Regression

epsilons=[1.35,1.5,1.75,1.9]
hr=HuberRegressor(alpha=0)

xx=np.linspace(X.min(), X.max(),7)

for e in epsilons:
    hr.epsilon=e
    hr.fit(X,y)
    y_pred=hr.coef_[0]*xx+hr.intercept_
    plt.plot(xx,y_pred, label='Hubber, e=%s' %(e))

plt.scatter(X.flatten(),y)
plt.legend()

# epsilon 값이 작을 때는 outlier를 무시함, epsilon값이 클때는 outlier에 반응하게 됨.

#%%
ridge=Ridge()
alpha=[0,0.5,1]

for a in alpha:
    ridge.alpha=a
    ridge.fit(X,y)
    y_pred=ridge.coef_[0]*xx+ridge.intercept_
    plt.plot(xx, y_pred, label="Ridge, alpha=%s"%(a))
    
plt.scatter(X.flatten(),y)
plt.legend()

# Ridge 함수는 아웃라이어 보다는 쓸모없는 변수에 집중해서 Robust(무시)함.
# 즉, 아웃라이어가 크게 벗어날때는 Ridge나 Huber epsilon을 작게 주어서 분석하는것이 유리.





