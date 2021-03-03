
from sklearn.datasets import make_classification
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVR

#%%

# Make dataset

b=0.1
w=1

x=np.linspace(-1,1,200)

x=np.random.rand(200)*2-1
y=w*x+b+np.random.randn(200)*0.2

plt.scatter(x,y)

#%%

# Fitting SVR

svr=SVR(kernel='linear', epsilon=0.1)
svr.fit(x[:, None],y)

svr.coef_
svr.intercept_
svr.dual_coef_
svr.n_support_

len(svr.dual_coef_[0])

svr.score(x.reshape(-1,1),y.reshape(-1,1)) # R-square

#%%

# Visualize

y_pred=svr.predict(x[:,None])

plt.scatter(x,y,c='blue')
plt.plot(x,y_pred,'r', label="SVR")
plt.legend(fontsize=14)

#%%

# NuSVR
# Compare SVR to NuSVR

from sklearn.svm import NuSVR

#%%

# Make Dataset

x=np.random.uniform(-3,4,100)
y=np.sin(x)+np.random.normal(size=100,scale=0.5)

#%%

# Fitting SVR

svr2=SVR(kernel='rbf', gamma=1, epsilon=0.1)
svr2.fit(x[:,None],y)

xx=np.linspace(-4,4,100)

yy=svr2.predict(xx[:,None])

# Fitting NuSVR

svr3 = NuSVR(kernel='rbf', gamma=1, nu=0.9)
svr3.fit(x[:,None],y)

yy2=svr3.predict(xx[:,None])

# Compare performance : R-square
svr2.score(x[:,None],y)
svr3.score(x[:,None],y)


#%%

# Visualize

plt.scatter(x,y)
plt.plot(xx,yy,'k',label='SVR')
plt.plot(xx,yy2,'r:', label="NuSVR")
plt.legend(fontsize=14)

#%%


