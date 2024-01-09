import pandas as pd
import os
import glob


print('Loading DataFrame......')
path = 'Data/FullData/LvSpeedingData'
path = os.path.join(path,'*.csv')
all_files = glob.glob(path)
data_df = pd.concat((pd.read_csv(f,usecols=['Event','Driver','Trip','Week']) for f in all_files))
#data_df = pd.read_csv('Data/FullData/TeenSpeedingData.csv',usecols=['Event','Driver','Trip','Week'])
se_df = pd.read_csv('Data/Events/StateFarm-SpeedingEvents/StateFarm-SpeedingEvents.csv')
summary_df = pd.read_csv('Data/Lv_KMeans/LvDisTime.csv',usecols=['Driver','Trip','SecInMotion'])
ndf = pd.DataFrame(columns=['Driver','Trip','Speeding','Total','Ratio'])

print('Adding frist two weeks......')
temp_df = data_df[data_df['Week']<3]
events = temp_df.Event.unique()
event_dict = {}
temp_df = se_df[se_df['Event'].isin(events)]
for index,row in temp_df.iterrows():
    key = (row['Driver'],row['Trip'])
    if key not in event_dict:
        event_dict[key] = (row['Speedingtime'],0)
    else:
        old = event_dict[key][0]
        new = old + row['Speedingtime']
        event_dict[key] = (new,0)

print('Adding last weeks......')
temp_df = data_df[data_df['Week']>=3]
events = temp_df.Event.unique()
temp_df = se_df[se_df['Event'].isin(events)]
for index,row in temp_df.iterrows():
    key = (row['Driver'],row['Trip'])
    if key not in event_dict:
        event_dict[key] = (row['Speedingtime'],1)
    else:
        old = event_dict[key][0]
        new = old + row['Speedingtime']
        event_dict[key] = (new,1)

print('Generating CSV file.....')
for key,value in event_dict.items():
    rows = summary_df[(summary_df['Driver']==key[0]) & (summary_df['Trip']==key[1])]
    total_time = 0
    for index,row in rows.iterrows():
        total_time += row['SecInMotion']
    total_time *= 100
    if total_time > 0:
        ratio = value[0]/total_time
        ndf = ndf.append({'Driver':key[0],'Trip':key[1],'Speeding':value[0],'Total':total_time,'Ratio':ratio}, ignore_index=True)
ndf = ndf.sort_values(by=['Driver','Trip'])
ndf = ndf.astype({'Driver':int,'Trip':int,'Speeding':int,'Total':int,'Ratio':float})
ndf.to_csv('Data/Lv_KMeans/Lv_K_means_Speeding.csv',index=False)

print('Done.')