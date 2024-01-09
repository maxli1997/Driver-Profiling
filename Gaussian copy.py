import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import r2_score
from scipy.stats import multivariate_normal
from scipy.stats import norm
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KernelDensity
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score

# Define simple gaussian
def gauss_function(x, amp, x0, sigma):
    return amp * np.exp(-(x - x0) ** 2. / (2. * sigma ** 2.))
#df = pd.read_csv('Data/KMeans/K_means_result.csv')
df = pd.read_csv('Setup_2/kmeans_all.csv')
dis_df1 = pd.read_csv('Data/KMeans/TeenDisTime.csv')
dis_df2 = pd.read_csv('Data/Lv_KMeans/LvDisTime.csv')
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
        rows = dis_df1[(dis_df1['Driver']==driver) & (dis_df1['Trip']==trip)]
        for i,rr in rows.iterrows():
            distance += rr['Distance']*0.001*0.62137
        rows = dis_df2[(dis_df2['Driver']*100==driver) & (dis_df2['Trip']==trip)]
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

'''# bic aic
bic = {}
aic = {}
for i in range(1,16):
    test_gmm = GaussianMixture(n_components=i,tol=1e-10,max_iter=1000,n_init=10)
    test_gmm = test_gmm.fit(rates)
    bic[i] = test_gmm.bic(rates)
    aic[i] = test_gmm.aic(rates)
plt.figure()
plt.plot(list(bic.keys()), list(bic.values()),color='blue',label='Bayesian Information Criterion')
plt.plot(list(aic.keys()), list(aic.values()),color='red',label='Akaike Information Criterion')
plt.xlabel("Number of Gaussian Components")
plt.ylabel("AIC/BIC")
plt.legend()
plt.show()
quit(0)'''


'''x_plot = np.linspace(-0.1, 1, 1000)
fig,ax =plt.subplots()
colors = ['navy','cornflowerblue','darkorange']
kernels = ['gaussian','tophat','epanechnikov']
lw = 2
std = np.std(rates)
h = 1.06*std*np.power(len(rates),-0.2)
for color,kernel in zip(colors,kernels):
    kde = KernelDensity(kernel=kernel,bandwidth=h).fit(rates)
    log_dens = kde.score_samples(x_plot.reshape(-1,1))
    ax.plot(x_plot,np.exp(log_dens),color=color,lw=lw,label="kernel = {0}".format(kernel))
ax.hist(rates,bins=10,density=1,color='skyblue',edgecolor='steelblue')
ax.legend(loc='upper right')
ax.set_ylabel("Density")
ax.set_xlabel("Risky Event Ratio")
ax.set_title("Kernel Density Estimation versus Histogram")
plt.show()
quit()'''

'''# Choose K
sil = {}
dav = {}
cal = {}
for k in range(2, 10):
    gmm = GaussianMixture(n_components=k,tol=1e-10,max_iter=1000,n_init=10).fit(rates)
    labels = gmm.predict(rates)
    sil[k] = silhouette_score(rates,labels)
    dav[k] = davies_bouldin_score(rates,labels)
    cal[k] = calinski_harabasz_score(rates,labels)
fig,ax = plt.subplots(1,3,figsize=[19.2,4.8])
ax[0].plot(list(sil.keys()), list(sil.values()))
ax[0].set(xlabel="Number of cluster",ylabel="Silhouette Score")
ax[0].set_title("Silhouette Score")
ax[1].plot(list(dav.keys()), list(dav.values()))
ax[1].set(xlabel="Number of cluster",ylabel="Davies_Bouldin Score")
ax[1].set_title("Davies_Bouldin Score")
ax[2].plot(list(cal.keys()), list(cal.values()))
ax[2].set(xlabel="Number of cluster",ylabel="Calinski_Harabasz Score")
ax[2].set_title("Calinski_Harabasz Score")
plt.show()
quit()'''

'''gmm = GaussianMixture(n_components=2,tol=1e-10,max_iter=1000,n_init=10)
gmm = gmm.fit(rates)
pp = gmm.predict(rates)
means = gmm.means_
for i in range(0,2):
    print(len(pp[pp==i]),means[i])
gmm_x = np.linspace(-0.1, 1, 10000)
gmm_y = np.exp(gmm.score_samples(gmm_x.reshape(-1, 1)))

bgm = BayesianGaussianMixture(n_components=4,tol=1e-10,n_init=10,max_iter=1000).fit(rates)
pp = bgm.predict(rates)
means = bgm.means_
for i in range(0,2):
    print(len(pp[pp==i]),means[i])
p_x = np.linspace(-0.1, 1, 1000)
p_y = np.exp(bgm.score_samples(p_x.reshape(-1, 1)))
plt.figure()
plt.plot(gmm_x, gmm_y, color="green", lw=2, label="Gaussian Mixture Model")
plt.plot(p_x, p_y, color="red", lw=2, label="Bayesian Gaussian Mixture Model")
plt.hist(rates,bins=10,density=1,color='skyblue',edgecolor='steelblue')
plt.legend(loc='upper right')
plt.ylabel("Probability density")
plt.xlabel("Risky Event Ratio")
plt.show()
quit()'''

'''# Choose K
sil = {}
dav = {}
cal = {}
for k in range(2, 10):
    gmm = GaussianMixture(n_components=k,tol=1e-10,max_iter=1000,n_init=10).fit(rates)
    labels = gmm.predict(rates)
    sil[k] = silhouette_score(rates,labels)
    dav[k] = davies_bouldin_score(rates,labels)
    cal[k] = calinski_harabasz_score(rates,labels)
bgm = BayesianGaussianMixture(n_components=4,tol=1e-10,max_iter=1000,n_init=10).fit(rates)
b_labels = bgm.predict(rates)
b_sil = silhouette_score(rates,b_labels)
b_dav = davies_bouldin_score(rates,b_labels)
b_cal = calinski_harabasz_score(rates,b_labels)
fig,ax = plt.subplots(1,3,figsize=[19.2,4.8])
ax[0].plot(list(sil.keys()), list(sil.values()))
ax[0].set(xlabel="Number of cluster",ylabel="Silhouette Score")
ax[0].set_title("Silhouette Score")
ax[0].plot(2,b_sil,'ro')
ax[1].plot(list(dav.keys()), list(dav.values()))
ax[1].set(xlabel="Number of cluster",ylabel="Davies_Bouldin Score")
ax[1].set_title("Davies_Bouldin Score")
ax[1].plot(2,b_dav,'ro')
ax[2].plot(list(cal.keys()), list(cal.values()))
ax[2].set(xlabel="Number of cluster",ylabel="Calinski_Harabasz Score")
ax[2].set_title("Calinski_Harabasz Score")
ax[2].plot(2,b_cal,'ro')
plt.show()
quit()'''

#X = rates[:,0]+rates[:,1]
gmm = GaussianMixture(n_components=3,tol=1e-10,max_iter=1000,n_init=10)
gmm = gmm.fit(rates)
bgm = BayesianGaussianMixture(n_components=4,tol=1e-10,n_init=10,max_iter=1000).fit(rates)
gmm = bgm

#gmm = gmm.fit(np.expand_dims(X, 1))
#pred = gmm.predict(np.expand_dims(X,1))
pred = gmm.predict(rates)
means = gmm.means_
#print(gmm.bic(rates))
for i in range(0,2):
    print(len(pred[pred==i]),means[i])

# Evaluate GMM
gmm_x = np.linspace(-0.1, 1, 10000)
gmm_y = np.exp(gmm.score_samples(gmm_x.reshape(-1, 1)))

group = np.array([gmm.means_.ravel(), gmm.covariances_.ravel(), gmm.weights_.ravel()])
group = np.transpose(group)
group = group[0:2,:]
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

result_list = [dr,label,total_trips,distances,rates.ravel()]
result = pd.DataFrame(result_list).transpose()
result.columns = ['Driver','Risk Level','TripsPerWeek','DistancePerTrip','RiskyEventsRate']
result.to_csv('Setup_2/BGM_all_3.csv',index=False)
#print (result)

print('Predicting......')
df = pd.read_csv('Setup_2/kmeans_pred_all.csv')
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
        rows = dis_df1[(dis_df1['Driver']==driver) & (dis_df1['Trip']==trip)]
        for i,rr in rows.iterrows():
            distance += rr['Distance']*0.001*0.62137
        rows = dis_df2[(dis_df2['Driver']*100==driver) & (dis_df2['Trip']==trip)]
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
    if high_rate >= th1:
        label.append(1)
    else:
        label.append(0)

print (totals)
dr = np.array(dr)
distances = np.array(distances)
total_trips = np.array(total_trips)
label = np.array(label)
print(len(label[label==0]))
print(len(label[label==1]))
ax[1].hist(rate,bins=10,density=0,color='skyblue',edgecolor='steelblue')
ax[1].axvline(x=th1,linestyle='--',c='red')
ax[1].text(0.26,4,'Threshold = '+str(th1),rotation=0,c='red')
ax[1].set_ylabel("Driver Count")
ax[1].set_xlabel("Risky Event Ratio")
plt.show()

result_list = [dr,label,total_trips,distances,np.array(rate)]
result = pd.DataFrame(result_list).transpose()
result.columns = ['Driver','Risk Level','TripsPerWeek','DistancePerTrip','RiskyEventsRate']
result.to_csv('Setup_2/BGM_pred_all_3.csv',index=False)

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


