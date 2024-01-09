import pandas as pd
import os
import glob

print('Tailgating')
df = pd.read_csv('Data/FullData/LvTailgatingData.csv')
events = df.Event.unique()
print('Adult: '+str(len(events)))
df = pd.read_csv('Data/FullData/TeenTailgatingData.csv')
events = df.Event.unique()
print('Teen: '+str(len(events)))

print('Lanechange')
path = 'Data/FullData/LvLanechangeData'
path = os.path.join(path,'*.csv')
all_files = glob.glob(path)
df = pd.concat((pd.read_csv(f) for f in all_files))
events = df.Event.unique()
print('Adult: '+str(len(events)))
df = pd.read_csv('Data/FullData/TeenLanechangeData.csv')
events = df.Event.unique()
print('Teen: '+str(len(events)))