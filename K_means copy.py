import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.cluster import OPTICS

print('Loading DataFrame......')
tailgating = pd.read_csv('Data/KMeans/K_means_Tailgating.csv')
hardbraking = pd.read_csv('Data/KMeans/K_means_Hardbraking.csv')
speeding = pd.read_csv('Data/KMeans/K_means_Speeding.csv')
lanechange = pd.read_csv('Data/KMeans/K_means_Lanechange.csv')
summary = pd.read_csv('Data/KMeans/TeenTripWeek.csv')
lv_tailgating = pd.read_csv('Data/Lv_KMeans/Lv_K_means_Tailgating.csv')
lv_hardbraking = pd.read_csv('Data/Lv_KMeans/Lv_K_means_Hardbraking.csv')
lv_speeding = pd.read_csv('Data/Lv_KMeans/Lv_K_means_Speeding.csv')
lv_lanechange = pd.read_csv('Data/Lv_KMeans/Lv_K_means_Lanechange.csv')
lv_summary = pd.read_csv('Data/Lv_KMeans/LvTripWeek.csv')
result = pd.DataFrame(columns=['Driver','Trip','Week','AgeGroup','Cluster'])

print('Creating Input Array......')
dataset = []
indexset = []
for index,row in summary.iterrows():
    if (row['Week'] >= 3):
        continue
    vector = []
    driver = row['Driver']
    trip = row['Trip']
    vector.append(driver)
    vector.append(trip)
    vector.append(row['Week'])
    flag = 0
    if ((tailgating['Driver'] == driver) & (tailgating['Trip'] == trip)).any():
        rows = tailgating[(tailgating['Driver'] == driver) & (tailgating['Trip'] == trip)]
        for i,r in rows.iterrows():
            vector.append(r['Ratio'])
        flag = 1
    else:
        vector.append(0)
    if ((speeding['Driver'] == driver) & (speeding['Trip'] == trip)).any():
        rows = speeding[(speeding['Driver'] == driver) & (speeding['Trip'] == trip)]
        for i,r in rows.iterrows():
            vector.append(r['Ratio'])
        flag = 1
    else:
        vector.append(0)
    if ((lanechange['Driver'] == driver) & (lanechange['Trip'] == trip)).any():
        rows = lanechange[(lanechange['Driver'] == driver) & (lanechange['Trip'] == trip)]
        for i,r in rows.iterrows():
            vector.append(r['Frequency'])
        flag = 1
    else:
        vector.append(0)
    if ((hardbraking['Driver'] == driver) & (hardbraking['Trip'] == trip)).any():
        rows = hardbraking[(hardbraking['Driver'] == driver) & (hardbraking['Trip'] == trip)]
        for i,r in rows.iterrows():
            '''if r['Frequency'] > 5:
                flag =2
                continue'''
            vector.append(r['Frequency'])
            flag = 1
    else:
        vector.append(0)
    if flag ==1 or flag==0 :
        dataset.append(vector)
        indexset.append([driver,trip,row['Week'],0])
for index,row in lv_summary.iterrows():
    if (row['Week'] >= 3):
        continue
    vector = []
    driver = row['Driver']
    trip = row['Trip']
    vector.append(driver)
    vector.append(trip)
    vector.append(row['Week'])
    flag = 0
    if ((lv_tailgating['Driver'] == driver) & (lv_tailgating['Trip'] == trip)).any():
        rows = lv_tailgating[(lv_tailgating['Driver'] == driver) & (lv_tailgating['Trip'] == trip)]
        for i,r in rows.iterrows():
            vector.append(r['Ratio'])
        flag = 1
    else:
        vector.append(0)
    if ((lv_speeding['Driver'] == driver) & (lv_speeding['Trip'] == trip)).any():
        rows = lv_speeding[(lv_speeding['Driver'] == driver) & (lv_speeding['Trip'] == trip)]
        for i,r in rows.iterrows():
            vector.append(r['Ratio'])
        flag = 1
    else:
        vector.append(0)
    if ((lv_lanechange['Driver'] == driver) & (lv_lanechange['Trip'] == trip)).any():
        rows = lv_lanechange[(lv_lanechange['Driver'] == driver) & (lv_lanechange['Trip'] == trip)]
        for i,r in rows.iterrows():
            vector.append(r['Frequency'])
        flag = 1
    else:
        vector.append(0)
    if ((lv_hardbraking['Driver'] == driver) & (lv_hardbraking['Trip'] == trip)).any():
        rows = lv_hardbraking[(lv_hardbraking['Driver'] == driver) & (lv_hardbraking['Trip'] == trip)]
        for i,r in rows.iterrows():
            vector.append(r['Frequency'])
            flag = 1
    else:
        vector.append(0)
    if flag ==1 or flag==0 :
        dataset.append(vector)
        indexset.append([driver*100,trip,row['Week'],row['AgeGroup']])
dataset = np.array(dataset)
data = dataset[:,3:]
raw_data = data.copy()
mean = np.mean(data, axis=0)
std = np.std(data, axis=0)
min_val = np.min(data,axis=0)
max_val = np.max(data,axis=0)
for i in range(0,len(mean)):
    data[:,i] = (data[:,i]-min_val[i])/(max_val[i]-min_val[i])

'''# Choose K
sse = {}
sil = {}
dav = {}
cal = {}
for k in range(2, 15):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(data)
    labels = kmeans.labels_
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
    sil[k] = silhouette_score(data,labels)
    dav[k] = davies_bouldin_score(data,labels)
    cal[k] = calinski_harabasz_score(data,labels)
fig,ax = plt.subplots(2,2,figsize=[12.8,9.6])
ax[0,0].plot(list(sse.keys()), list(sse.values()))
ax[0,0].set(xlabel="Number of cluster",ylabel="SSE")
ax[0,0].set_title("Sum of Squared Error")
ax[0,1].plot(list(sil.keys()), list(sil.values()))
ax[0,1].set(xlabel="Number of cluster",ylabel="Silhouette Score")
ax[0,1].set_title("Silhouette Score")
ax[1,0].plot(list(dav.keys()), list(dav.values()))
ax[1,0].set(xlabel="Number of cluster",ylabel="Davies_Bouldin Score")
ax[1,0].set_title("Davies_Bouldin Score")
ax[1,1].plot(list(cal.keys()), list(cal.values()))
ax[1,1].set(xlabel="Number of cluster",ylabel="Calinski_Harabasz Score")
ax[1,1].set_title("Calinski_Harabasz Score")
plt.savefig('kmeans_metrics.png')
quit(0)'''

print('K-means Clustering.....')
kmeans = KMeans(n_clusters=2,tol=1e-10 ,max_iter=10000, n_init=10).fit(data)
for i in range(len(indexset)):
    row = indexset[i]
    result = result.append({'Driver':row[0],'Trip':row[1],'Week':row[2],'AgeGroup':row[3],'Cluster':kmeans.labels_[i]}, ignore_index=True)
result = result.astype({'Driver':int,'Trip':int,'Week':int,'AgeGroup':int,'Cluster':int})
result.to_csv('setup_2/kmeans_all.csv',index=False)

print('Plotting......')
label0 = raw_data[kmeans.labels_ == 0]
avg0 = np.mean(label0,axis=0)
label1 = raw_data[kmeans.labels_ == 1]
avg1 = np.mean(label1,axis=0)
print(len(label0),len(label1))
print(avg0)
print(avg1)

fig = plt.figure(figsize=(12.8,4.8))
ax = fig.add_subplot(121,projection='3d')
ax.scatter(label0[:,0],label0[:,1],label0[:,2],s=label0[:,3]*10+30,c='skyblue',marker='+')
ax.scatter(label1[:,0],label1[:,1],label1[:,2],s=label1[:,3]*10+30,c='orange',marker='+')
ax.set_xlabel('Tailgating Ratio')
ax.set_ylabel('Speeding Ratio')
ax.set_zlabel('Lanechange count/mile')

print('Predicting......')
dataset = []
indexset = []
for index,row in summary.iterrows():
    if (row['Week'] <= 3 or row['Week'] > 7):
        continue
    vector = []
    driver = row['Driver']
    trip = row['Trip']
    vector.append(driver)
    vector.append(trip)
    vector.append(row['Week'])
    flag = 0
    if ((tailgating['Driver'] == driver) & (tailgating['Trip'] == trip)).any():
        rows = tailgating[(tailgating['Driver'] == driver) & (tailgating['Trip'] == trip)]
        for i,r in rows.iterrows():
            vector.append(r['Ratio'])
        flag = 1
    else:
        vector.append(0)
    if ((speeding['Driver'] == driver) & (speeding['Trip'] == trip)).any():
        rows = speeding[(speeding['Driver'] == driver) & (speeding['Trip'] == trip)]
        for i,r in rows.iterrows():
            vector.append(r['Ratio'])
        flag = 1
    else:
        vector.append(0)
    if ((lanechange['Driver'] == driver) & (lanechange['Trip'] == trip)).any():
        rows = lanechange[(lanechange['Driver'] == driver) & (lanechange['Trip'] == trip)]
        for i,r in rows.iterrows():
            vector.append(r['Frequency'])
        flag = 1
    else:
        vector.append(0)
    if ((hardbraking['Driver'] == driver) & (hardbraking['Trip'] == trip)).any():
        rows = hardbraking[(hardbraking['Driver'] == driver) & (hardbraking['Trip'] == trip)]
        for i,r in rows.iterrows():
            vector.append(r['Frequency'])
        flag = 1
    else:
        vector.append(0)
    if flag ==1 or flag==0 :
        dataset.append(vector)
        indexset.append([driver,trip,row['Week'],0])
for index,row in lv_summary.iterrows():
    if (row['Week'] < 3 or row['Week'] >= 7):
        continue
    vector = []
    driver = row['Driver']
    trip = row['Trip']
    vector.append(driver)
    vector.append(trip)
    vector.append(row['Week'])
    flag = 0
    if ((lv_tailgating['Driver'] == driver) & (lv_tailgating['Trip'] == trip)).any():
        rows = lv_tailgating[(lv_tailgating['Driver'] == driver) & (lv_tailgating['Trip'] == trip)]
        for i,r in rows.iterrows():
            vector.append(r['Ratio'])
        flag = 1
    else:
        vector.append(0)
    if ((lv_speeding['Driver'] == driver) & (lv_speeding['Trip'] == trip)).any():
        rows = lv_speeding[(lv_speeding['Driver'] == driver) & (lv_speeding['Trip'] == trip)]
        for i,r in rows.iterrows():
            vector.append(r['Ratio'])
        flag = 1
    else:
        vector.append(0)
    if ((lv_lanechange['Driver'] == driver) & (lv_lanechange['Trip'] == trip)).any():
        rows = lv_lanechange[(lv_lanechange['Driver'] == driver) & (lv_lanechange['Trip'] == trip)]
        for i,r in rows.iterrows():
            vector.append(r['Frequency'])
        flag = 1
    else:
        vector.append(0)
    if ((lv_hardbraking['Driver'] == driver) & (lv_hardbraking['Trip'] == trip)).any():
        rows = lv_hardbraking[(lv_hardbraking['Driver'] == driver) & (lv_hardbraking['Trip'] == trip)]
        for i,r in rows.iterrows():
            vector.append(r['Frequency'])
            flag = 1
    else:
        vector.append(0)
    if flag ==1 or flag==0 :
        dataset.append(vector)
        indexset.append([driver*100,trip,row['Week'],row['AgeGroup']])

#for dat in dataset:

dataset = np.array(dataset)
data = dataset[:,3:]
raw_data = data.copy()
mean = np.mean(data, axis=0)
min_val = np.min(data,axis=0)
max_val = np.max(data,axis=0)
std = np.std(data, axis=0)
for i in range(0,len(mean)):
    data[:,i] = (data[:,i]-min_val[i])/(max_val[i]-min_val[i])

prediction = kmeans.predict(data)
result = pd.DataFrame(columns=['Driver','Trip','Week','AgeGroup','Cluster'])
for i in range(len(indexset)):
    row = indexset[i]
    result = result.append({'Driver':row[0],'Trip':row[1],'Week':row[2],'AgeGroup':row[3],'Cluster':prediction[i]}, ignore_index=True)
result = result.astype({'Driver':int,'Trip':int,'Week':int,'AgeGroup':int,'Cluster':int})
result.to_csv('setup_2/kmeans_pred_all.csv',index=False)

print('Plotting......')
label0 = raw_data[prediction == 0]
avg0 = np.mean(label0,axis=0)
label1 = raw_data[prediction == 1]
avg1 = np.mean(label1,axis=0)
print(len(label0),len(label1))
print(avg0)
print(avg1)

bx = fig.add_subplot(122,projection='3d')
bx.scatter(label0[:,0],label0[:,1],label0[:,2],s=label0[:,3]*10+30,c='skyblue',marker='+')
bx.scatter(label1[:,0],label1[:,1],label1[:,2],s=label1[:,3]*10+30,c='orange',marker='+')
bx.set_xlabel('Tailgating Ratio')
bx.set_ylabel('Speeding Ratio')
bx.set_zlabel('Lanechange count/mile')
plt.show()

print('Done.')