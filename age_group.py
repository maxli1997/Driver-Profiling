import pandas as pd
import glob
import os

path = 'Data/FullData/LvSpeedingData'
path = os.path.join(path,'*.csv')
all_files = glob.glob(path)
df = pd.concat((pd.read_csv(f) for f in all_files))
agegroups = df.AgeGroup.unique()
print('Speeding:')
for agegroup in agegroups:
    temp_df = df[df['AgeGroup']==agegroup]
    temp_df = temp_df.Event.unique()
    print(str(agegroup)+': '+str(len(temp_df)))
        
path = 'Data/FullData/LvLanechangeData'
path = os.path.join(path,'*.csv')
all_files = glob.glob(path)
df = pd.concat((pd.read_csv(f) for f in all_files))
agegroups = df.AgeGroup.unique()
print('Lanechange:')
for agegroup in agegroups:
    temp_df = df[df['AgeGroup']==agegroup]
    temp_df = temp_df.Event.unique()
    print(str(agegroup)+': '+str(len(temp_df)))

df = pd.read_csv('Data/FullData/LvTailgatingData.csv')
agegroups = df.AgeGroup.unique()
print('Tailgating:')
for agegroup in agegroups:
    temp_df = df[df['AgeGroup']==agegroup]
    temp_df = temp_df.Event.unique()
    print(str(agegroup)+': '+str(len(temp_df)))

df = pd.read_csv('Data/FullData/LvHardbrakingData.csv')
agegroups = df.AgeGroup.unique()
print('Hardbraking:')
for agegroup in agegroups:
    temp_df = df[df['AgeGroup']==agegroup]
    temp_df = temp_df.Event.unique()
    print(str(agegroup)+': '+str(len(temp_df)))