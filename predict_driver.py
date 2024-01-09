import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import r2_score
from scipy.stats import multivariate_normal
from scipy.stats import norm

df = pd.read_csv('KmeansTeenWeek4567.csv')
#df = df[df['AgeGroup'] == 2]
dis_df = pd.read_csv('Data/KMeans/TeenDisTime.csv')
drivers = df.Driver.unique()
driver_num = len(drivers)
th1 = 0.25577558
#th2 = 0.20891089
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
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[8, 5])
ax.hist(rate,bins=10,density=0,color='skyblue',edgecolor='steelblue')
plt.axvline(x=0.25577558,linestyle='--',c='red')
plt.text(0.26,4,'Threshold = 0.25577558',rotation=0,c='red')
ax.set_ylabel("Driver Count")
ax.set_xlabel("Risky Event Ratio")
plt.show()

result_list = [dr,label,total_trips,distances]
result = pd.DataFrame(result_list).transpose()
result.columns = ['Driver','Risk Level','TripsPerWeek','DistancePerTrip']
result.to_csv('GMMTeen47.csv',index=False)