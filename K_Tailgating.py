import pandas as pd


print('Loading DataFrame......')
data_df = pd.read_csv('Data/FullData/TeenTailgatingData.csv',usecols=['Event','Driver','Trip','Week'])
event_df = pd.read_csv('Data/Events/StateFarm-TailgatingTeenEvents.csv')
summary_df = pd.read_csv('Data/KMeans/TeenDisTime.csv',usecols=['Driver','Trip','TripTime'])
ndf = pd.DataFrame(columns=['Driver','Trip','Tailgating','Total','Ratio'])

print('Adding frist two weeks......')
temp_df = data_df[data_df['Week']<3]
events = temp_df.Event.unique()
event_dict = {}
temp_df = event_df[event_df['Event'].isin(events)]
for index,row in temp_df.iterrows():
    key = (row['Driver'],row['Trip'])
    if key not in event_dict:
        event_dict[key] = (row['endtime']-row['starttime'],0)
    else:
        old = event_dict[key][0]
        new = old + row['endtime']-row['starttime']
        event_dict[key] = (new,0)

print('Adding last weeks......')
temp_df = data_df[data_df['Week']>=3]
events = temp_df.Event.unique()
temp_df = event_df[event_df['Event'].isin(events)]
for index,row in temp_df.iterrows():
    key = (row['Driver'],row['Trip'])
    if key not in event_dict:
        event_dict[key] = (row['endtime']-row['starttime'],1)
    else:
        old = event_dict[key][0]
        new = old + row['endtime']-row['starttime']
        event_dict[key] = (new,1)

print('Generating CSV file.....')
for key,value in event_dict.items():
    rows = summary_df[(summary_df['Driver']==key[0]) & (summary_df['Trip']==key[1])]
    total_time = 0
    for index,row in rows.iterrows():
        total_time += row['TripTime']
    ratio = value[0]/total_time
    ndf = ndf.append({'Driver':key[0],'Trip':key[1],'Tailgating':value[0],'Total':total_time,'Ratio':ratio}, ignore_index=True)
ndf = ndf.sort_values(by=['Driver','Trip'])
ndf = ndf.astype({'Driver':int,'Trip':int,'Tailgating':int,'Total':int,'Ratio':float})
ndf.to_csv('Data/KMeans/K_means_Tailgating.csv',index=False)

print('Done.')