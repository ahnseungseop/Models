# 자체 생성 데이터로 DBSCAN  실험
험
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN


X1,y1 = datasets.make_moons(n_samples=200, noise=0.1, random_state=10)

plt.scatter(X1[:,0], X1[:,1], c=y1)

X2,y2=datasets.make_circles(n_samples=200, noise=0.1, factor=0.5)

fig=plt.figure(figsize=(6,6))
plt.scatter(X2[:,0],X2[:,1], c=y2)

dbscan=DBSCAN(eps=0.3, min_samples=20)
dbscan.fit(X1)

cores=dbscan.core_sample_indices_

y1_1 = dbscan.labels_ #-1은 아웃라이어

plt.scatter(X1[y1_1!=-1,0], X1[y1_1!=-1,1], c=y1_1[y1_1!=-1])
plt.scatter(X1[y1_1==-1,0], X1[y1_1==-1,1], marker='x',c='k')

dbscan=DBSCAN(eps=0.15, min_samples=3)
dbscan.fit(X2)
y1_2=dbscan.labels_

plt.scatter(X2[y1_2!=-1,0], X2[y1_2!=-1,1], c=y1_2[y1_2!=-1])
plt.scatter(X2[y1_2==-1,0], X2[y1_2==-1,1], marker='x',c='k')

# 밀도가 낮은 곳에 위치한 지점들이 아웃라이어로 설정이 됨


# gmap과 DBSCAN을 사용하여 지도상에 범죄다발지역 표시 


import pandas as pd

crime=pd.read_excel('File_location/crime.xlsx')

crime_sel = crime.head(100)
crime['Primary Type'].value_counts()

# 어느 지역에 어느 범죄가 많이 일어나나?
# 지도위에 좌표를 찍는것이 도움이 됨.
from gmplot import gmplot
import numpy as np

# 지도위에 강도 발생 지점표시 (10,000개 발생 지점만)

crime_type="ROBBERY"
gmap=gmplot.GoogleMapPlotter(41.832621,-87.658502,11)
X=crime[crime['Primary Type']==crime_type][['Latitude','Longitude']].values
if len(X)>10000:
    ind=np.random.choice(range(len(X)), size=10000, replace=False)
else :
    ind=range(len(X))
    
gmap.scatter(X[ind,0], X[ind,1], size=40, marker=False)
gmap.heatmap(X[ind,0], X[ind,1])
gmap.draw(r'File_location/%s.html'%(crime_type.replace(' ','_')))

# DBSCAN 활용하여 지도 상에 표시

unit=3280.84

X2=crime[crime['Primary Type']==crime_type][['X Coordinate', 'Y Coordinate']].values
X2=X2[ind]
dbscan=DBSCAN(eps=0.5*unit, min_samples=50)
dbscan.fit(X2)

label=dbscan.labels_
np.unique(label)

sel_ind = ind[label!=-1]

from matplotlib.colors import to_hex

colors = plt.cm.hsv(label[label!=-1]/label.max())
colors_hex=[to_hex(c) for c in colors ]

gmap=gmplot.GoogleMapPlotter(41.832621,-87.658502,11)
gmap.scatter(X[sel_ind,0], X[sel_ind,1], size=40, marker=False, color=colors_hex)
gmap.heatmap(X[sel_ind,0], X[sel_ind,1])
gmap.draw(r'File_location/%s_CLS.html'%(crime_type.replace(' ','_')))




