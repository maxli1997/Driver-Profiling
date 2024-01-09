import pandas as pd
import glob
import os

path = 'Data/FullData/LvSpeedingData'
path = os.path.join(path,'*.csv')
all_files = glob.glob(path)


df = pd.concat((pd.read_csv(f) for f in all_files))
ndf = pd.DataFrame(columns=['Event','AvgSpeed','MaxSpeed'])
ndf = ndf.astype({'Event':int,'AvgSpeed':float,'MaxSpeed':float})

events = df.Event.unique()
for event in events:
    print(event)
    temp_df = df[df['Event']==event]
    avg = temp_df.Speed.mean()
    max_val = temp_df.Speed.max()
    ndf = ndf.append({'Event':event,'AvgSpeed':avg,'MaxSpeed':max_val}, ignore_index=True)

        
ndf.to_csv('lv_speeding_stat.csv',index=False)