import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import r2_score
from scipy.stats import multivariate_normal
from scipy.stats import norm

# Define simple gaussian
def gauss_function(x, amp, x0, sigma):
    return amp * np.exp(-(x - x0) ** 2. / (2. * sigma ** 2.))
#df = pd.read_csv('Data/KMeans/K_means_result.csv')
df = pd.read_csv('Setup_2/kmeans_teen.csv')
dis_df = pd.read_csv('Data/KMeans/TeenDisTime.csv')
drivers = df.Driver.unique()
driver_num = len(drivers)
rates = []
total_trips = []
distances = []
dr = []
for driver in drivers:
    temp_df = df[df['Driver']==driver]
    distance = 0
    for index,r in temp_df.iterrows():
        trip = r['Trip']
        rows = dis_df[(dis_df['Driver']==driver) & (dis_df['Trip']==trip)]
        for i,rr in rows.iterrows():
            distance += rr['Distance']*0.001*0.62137
    total = len(temp_df)
    if total < 5:
        continue
    #temp = temp_df[temp_df['Cluster']==2]
    #medium = len(temp)
    temp = temp_df[temp_df['Cluster']==1]
    high = len(temp)
    #medium_rate = medium/total
    high_rate = high/total
    total_trips.append(total/2)
    rates.append([high_rate])
    distances.append(distance/total)
    dr.append(driver)


# Fit GMM
dr = np.array(dr)
distances = np.array(distances)
total_trips = np.array(total_trips)
rates = np.array(rates)
#X = rates[:,0]+rates[:,1]
gmm = GaussianMixture(n_components=2,tol=1e-10,max_iter=1000,n_init=10)
gmm = gmm.fit(rates)
#gmm = gmm.fit(np.expand_dims(X, 1))
#pred = gmm.predict(np.expand_dims(X,1))
pred = gmm.predict(rates)
means = gmm.means_
print(gmm.bic(rates))
for i in range(0,2):
    print(len(pred[pred==i]),means[i])

# Evaluate GMM
gmm_x = np.linspace(-0.1, 1, 10000)
gmm_y = np.exp(gmm.score_samples(gmm_x.reshape(-1, 1)))

group = np.array([gmm.means_.ravel(), gmm.covariances_.ravel(), gmm.weights_.ravel()])
group = np.transpose(group)
sq = group[:,0].argsort()
group = group[sq]
print (group)
label = np.zeros(len(dr),int)
print(sq)
print(pred)
for i in range(len(sq)):
    for j in range(len(pred)):
        if pred[j]==sq[i]:
            label[j] = i


# Construct function manually as sum of gaussians
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=[12.8, 4.8])
ax[0].hist(rates,bins=10,density=1,color='skyblue',edgecolor='steelblue')
# ax.hist(X, bins=10,density=1)
ax[0].plot(gmm_x, gmm_y, color="black", lw=2, label="Fitted Curve")
colorgroup = ['forestgreen','yellow','red']
labels = ['conservative','risky','high risk']
gmm_ = []
for i in range(len(group)):
    gauss = gauss_function(x=gmm_x, amp=1, x0=group[i][0], sigma=np.sqrt(group[i][1]))
    gmm = (gauss / np.trapz(gauss, gmm_x)) * group[i][2]
    gmm_ .append(gmm)
    ax[0].plot(gmm_x, gmm, color=colorgroup[i], lw=2, label=labels[i], linestyle="dashed")
for i in range(len(group)-1):
    idx = np.argwhere(np.diff(np.sign(gmm_[i] - gmm_[i+1]))).flatten()
    plt.plot(gmm_x[idx], gmm_[i][idx], 'o',c='Orange')
#idx2 = np.argwhere(np.diff(np.sign(gmm_[1] - gmm_[2]))).flatten()
    print(gmm_x[idx])
#plt.plot(gmm_x[idx2[-1]], gmm_[1][idx2[-1]], 'o')
# Annotate diagram
ax[0].set_ylabel("Probability density")
ax[0].set_xlabel("Risky Event Ratio")

result_list = [dr,label,total_trips,distances]
result = pd.DataFrame(result_list).transpose()
result.columns = ['Driver','Risk Level','TripsPerWeek','DistancePerTrip']
result.to_csv('Setup_2/GMM_teen.csv',index=False)
#print (result)

print('Predicting......')
df = pd.read_csv('Setup_2/kmeans_pred_teen.csv')
drivers = df.Driver.unique()
driver_num = len(drivers)
th1 = gmm_x[idx][-1]
label = []
rate = []
total_trips = []
distances = []
totals = []
dr = []
for driver in drivers:
    temp_df = df[df['Driver']==driver]
    distance = 0
    for index,r in temp_df.iterrows():
        trip = r['Trip']
        rows = dis_df[(dis_df['Driver']==driver) & (dis_df['Trip']==trip)]
        for i,rr in rows.iterrows():
            distance += rr['Distance']*0.001*0.62137
    total = len(temp_df)
    if total < 5:
        continue
    totals.append(total)
    temp = temp_df[temp_df['Cluster']==1]
    high = len(temp)
    high_rate = high/total
    rate.append(high_rate)
    total_trips.append(total/4)
    distances.append(distance/total)
    dr.append(driver)
    if high_rate <= th1:
        label.append(0)
    #elif high_rate <= th2:
    #    label.append(1)
    else:
        label.append(1)

print (totals)
dr = np.array(dr)
distances = np.array(distances)
total_trips = np.array(total_trips)
label = np.array(label)
print(len(label[label==0]))
print(len(label[label==1]))
#print(len(label[label==2]))
ax[1].hist(rate,bins=10,density=0,color='skyblue',edgecolor='steelblue')
ax[1].axvline(x=th1,linestyle='--',c='red')
ax[1].text(0.26,4,'Threshold = '+str(th1),rotation=0,c='red')
ax[1].set_ylabel("Driver Count")
ax[1].set_xlabel("Risky Event Ratio")
plt.show()

result_list = [dr,label,total_trips,distances]
result = pd.DataFrame(result_list).transpose()
result.columns = ['Driver','Risk Level','TripsPerWeek','DistancePerTrip']
result.to_csv('Setup_2/GMM_pred_teen.csv',index=False)

'''

X=np.array(rates)
plt.axis([0, 0.4, 0, 0.4])
plt.scatter(X.T[0], X.T[1])
plt.show()

# Fit the data
gmm = GaussianMixture(n_components=3,tol=1e-10,max_iter=500,n_init=10)
gmm.fit(X)
print(gmm.bic(X))

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
lin_param = (0, 1, 1000)
x = np.linspace(*lin_param)
y = np.linspace(*lin_param)

xx, yy = np.meshgrid(x, y)
pos = np.concatenate((xx.reshape(-1, 1), yy.reshape(-1, 1)), axis = 1)
z = gmm.score_samples(pos) # Note that this method returns log-likehood
zz = z
#z = np.exp(gmm.score_samples(pos)) # e^x to get likehood values
z = z.reshape(xx.shape)
zz = zz.reshape(xx.shape)

# Construct function manually as sum of gaussians
#gmm_z_sum = []
#for m, c, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
#    pos = np.dstack((xx, yy))
#    rv = multivariate_normal(mean=m,cov=c)
#    ax.contourf(x, y, rv.pdf(pos),cmap='binary')

ax.contour3D(x, y, z, 50, cmap='viridis')

plt.show()
fig2 = plt.figure()
plt.contourf(x, y, zz, 50, cmap="viridis")
plt.show()

pred = gmm.predict(X)
print(len(pred[pred==0]),len(pred[pred==1]),len(pred[pred==2]))




'''


