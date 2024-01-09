import pandas as pd



df = pd.read_csv('Data/FullData/LvTailgatingHeadway.csv')
df = df.sort_values(by=['Event','Driver','Trip','Time'])
ndf = pd.DataFrame(columns=['Event','AvgHeadway'])
ndf = ndf.astype({'Event':int,'AvgHeadway':float})

events = df.Event.unique()
for event in events:
    print(event)
    temp_df = df[df['Event']==event]
    avg = temp_df.TimeHeadway.mean()
    ndf = ndf.append({'Event':event,'AvgHeadway':avg}, ignore_index=True)

        
ndf.to_csv('lv_headway_stat.csv',index=False)