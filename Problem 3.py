import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv('D:/IP_LP3_DATA_SCIENCE_DEBJYOTI_SAHA_2982/Week 1/DS_DATESET.csv')
print(data)

fig=plt.figure(figsize=(5,5))
x=data['Areas']
y=data['Label']
n=range(1,8)
fig, ax=plt.subplots()
ax.scatter(x,y,marker='o', c='red', alpha=0.5)
plt.grid()
for i, txt in enumerate(n):
    ax.annotate(txt, (x[i], y[i]))
plt.show()

from sklearn.cluster import KMeans
kmeans=KMeans()
print(kmeans.fit(data))

labels=kmeans.predict(data)
centroids=kmeans.cluster_centres_
print(centroids)

fig=plt.figure(figsize=(5,5))
colmap={1:'r', 2:'b'}
colors=map(lambda x: colmap[x+1], labels)
colors1= list(colors)
fig, ax=plt.subplots()
ax.scatter(x,y,color=colors1, alpha=0.5, edgecolor='k')
for idx, centroid in enumerate(centroids):
    plt.scatter(*centroid, color=colmap[idx+1])

for i, txt in enumerate(n):
    ax. annotate(txt, (x[i],y[i]))
plt.grid()
plt.xlim(0,8)
plt.ylim(0,8)
plt.show()
