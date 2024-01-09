import pandas as pd
import os
import glob

print('Loading DataFrame......')
'''path = 'Data/FullData/LvLanechangeData'
path = os.path.join(path,'*.csv')
all_files = glob.glob(path)
data_df = pd.concat((pd.read_csv(f,usecols=['Event','Driver','Trip','Week']) for f in all_files))'''
data_df = pd.read_csv('Data/FullData/TeenLanechangeData.csv',usecols=['Event','Driver','Trip','Week'])
event_df = pd.read_csv('Data/Events/StateFarm-LanechangeTeenEvents.csv')
summary_df = pd.read_csv('Data/KMeans/TeenDisTime.csv',usecols=['Driver','Trip','Distance'])
ndf = pd.DataFrame(columns=['Driver','Trip','Lanechange','Distance','Frequency'])

print('Adding frist two weeks......')
temp_df = data_df[data_df['Week']<3]
events = temp_df.Event.unique()
event_dict = {}
temp_df = event_df[event_df['Event'].isin(events)]
for index,row in temp_df.iterrows():
    key = (row['Driver'],row['Trip'])
    if key not in event_dict:
        event_dict[key] = (1,0)
    else:
        prev = event_dict[key][0]
        treat = event_dict[key][1]
        event_dict[key] = (prev+1,treat)

print('Adding last weeks......')
temp_df = data_df[data_df['Week']>=3]
events = temp_df.Event.unique()
temp_df = event_df[event_df['Event'].isin(events)]
for index,row in temp_df.iterrows():
    key = (row['Driver'],row['Trip'])
    if key not in event_dict:
        event_dict[key] = (1,1)
    else:
        prev = event_dict[key][0]
        treat = event_dict[key][1]
        event_dict[key] = (prev+1,treat)

print('Generating CSV file.....')
for key,value in event_dict.items():
    rows = summary_df[(summary_df['Driver']==key[0]) & (summary_df['Trip']==key[1])]
    distance = 0
    for index,row in rows.iterrows():
        distance += row['Distance']*0.001*0.62137
    freq = value[0]/distance
    ndf = ndf.append({'Driver':key[0],'Trip':key[1],'Lanechange':value[0],'Distance':distance,'Frequency':freq}, ignore_index=True)
ndf = ndf.sort_values(by=['Driver','Trip'])
ndf = ndf.astype({'Driver':int,'Trip':int,'Lanechange':int,'Distance':float,'Frequency':float})
ndf.to_csv('Data/KMeans/K_means_Lanechange.csv',index=False)

print('Done.')