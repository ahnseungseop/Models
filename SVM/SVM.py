# SVC(SVM CLASSIFICAION)
# Prameter
# C : sofr margine classification을 할때 error term에 대한 penalty default:1 / 크게 조절할수록 hard margine에 가까워짐
# kernal : default : rbf
# degree : polynominal kernal 함수의 승수
# gamma : default = 0 설정하지 않음 / 1/feature수 로 설정되어지게 됨. 
# coef0 : poly, sigmoid 뒤에 붙는 상수항
# probability : 확률 기반으로 변환해서 쓸 수 있는 알고리즘 

from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

X1, y1 = datasets.make_classification(n_samples=500, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=6)

plt.scatter(X1[:,0],X1[:,1],c=y1,alpha=0.5)

clf1=SVC(kernel='linear',C=0.01) 
clf1.fit(X1,y1)

clf1.support_ # 서포트 벡터의 목록(인덱스) 출력
clf1.dual_coef_
clf1.support_vectors_ #서포트 벡터의 값
clf1.n_support_ # 각 클래스별로 서포트벡터의 갯수 출력
clf1.coef_
clf1.intercept_

# Drawing decison boundary 
w=clf1.coef_[0]
xx=np.linspace(-3,4,100)
yy=-(w[0]/w[1])*xx-(clf1.intercept_[0]/w[1])

plt.scatter(X1[:,0],X1[:,1],c=y1,alpha=0.5)
plt.plot(xx,yy)

# Compare to logistic regression
from sklearn.linear_model import LogisticRegression
logistic=LogisticRegression()
logistic.fit(X1,y1)
logistic.coef_
logistic.intercept_

w2 = logistic.coef_[0]
yy2 = -(w2[0] /w2[1])*xx-(logistic.intercept_[0]/w2[1])
plt.scatter(X1[:,0],X1[:,1],c=y1,alpha=0.5)
plt.plot(xx,yy2)


# nonlinear
from sklearn import datasets
X2, y2 = datasets.make_moons(n_samples=200, noise=0.17, random_state=33)
plt.scatter(X2[:,0], X2[:,1], c=y2)

clf2 = SVC(kernel='rbf', gamma=1, C=1)
clf2.fit(X2,y2)

clf2.support_
clf2.n_support_
clf2.dual_coef_

X,Y =np.meshgrid(np.linspace(-2,3,100), np.linspace(-1,2,100))

Z=np.c_[X.ravel(),Y.ravel()]
Z_pred=clf2.predict(Z)

plt.scatter(X2[:,0], X2[:,1], c=y2)
plt.contour # 같은 높이에 해당하는 점들 
plt.contourf(X,Y,np.reshape(Z_pred, X.shape),alpha=0.5) # 같은 높이에 해당하는 점들을 다른 색깔로 표시


# Decision Boundary of Decison Tree 
from sklearn.tree import DecisionTreeClassifier
tr=DecisionTreeClassifier(max_depth=10)
tr.fit(X2,y2)
Z_pred_tr=tr.predict(Z)

plt.scatter(X2[:,0], X2[:,1], c=y2)
plt.contourf(X,Y,np.reshape(Z_pred_tr, X.shape),alpha=0.5)

# Decision Boundary of GaussianNB
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(X2,y2)

Z_pred_nb=nb.predict(Z)
plt.scatter(X2[:,0], X2[:,1], c=y2)
plt.contourf(X,Y,np.reshape(Z_pred_nb, X.shape),alpha=0.5)

