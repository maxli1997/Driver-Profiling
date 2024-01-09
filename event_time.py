import pandas as pd



df = pd.read_csv('Data/Brain-LaneChange.csv',usecols=['DriverID','Trip','Time','LC_SandE_flag'])
df = df[df['LC_SandE_flag']!=0]
df = df.sort_values(by=['DriverID','Trip','Time','LC_SandE_flag'])
ndf = pd.DataFrame(columns=['Event','Driver','Trip','Starttime','Endtime','Duration'])
ndf = ndf.astype({'Event':int,'Driver':int,'Trip':int,'Starttime':int,'Endtime':int,'Duration':float})

event = 0
cnt = 0
for index,row in df.iterrows():
    if cnt==0 and row['LC_SandE_flag']==1:
        event += 1
        driver = row['DriverID']
        trip = row['Trip']
        time = row['Time']
        flag = df['LC_SandE_flag']
        starttime = time
        cnt = 1
    elif cnt==1 and row['LC_SandE_flag']==2:
        time = row['Time']
        endtime = time
        duration = (endtime-starttime)/100
        ndf = ndf.append({'Event':event,'Driver':driver,'Trip':trip,'Starttime':starttime,'Endtime':endtime,'Duration':duration}, ignore_index=True)
        #write row
        cnt = 0
    prev_row = row
        
ndf.to_csv('result.csv',index=False)