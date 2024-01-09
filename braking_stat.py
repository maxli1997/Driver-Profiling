import pandas as pd



df = pd.read_csv('Data/FullData/TeenHardbrakingData.csv')
df = df.sort_values(by=['Event','Driver','Trip','Time'])
ndf = pd.DataFrame(columns=['Event','AvgAx','MinAx'])
ndf = ndf.astype({'Event':int,'AvgAx':float,'MinAx':float})

events = df.Event.unique()
for event in events:
    print(event)
    temp_df = df[df['Event']==event]
    avg = temp_df.Ax.mean()
    min_val = temp_df.Ax.min()
    ndf = ndf.append({'Event':event,'AvgAx':avg,'MinAx':min_val}, ignore_index=True)

        
ndf.to_csv('teen_braking_stat.csv',index=False)