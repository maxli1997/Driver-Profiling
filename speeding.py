import pandas as pd



df = pd.read_csv('Data/StateFarm-Speeding(UNCLEANED).csv')
ndf = pd.DataFrame(columns=['Event','Driver','Trip','Starttime','Endtime','Speedingtime'])

event = 1
driver = df['Driver'][0]
trip = df['Trip'][0]
time = df['Time'][0]
duration = 10
starttime = time
endtime = time
for index,row in df.iterrows():
    if row['Driver'] == driver and row['Trip'] == trip and row['Time'] == time:
        continue
    elif row['Driver'] == driver and row['Trip'] == trip and abs(row['Time'] - time) <1000:
        time = row['Time']
        duration += 10
        endtime = time
    else:
        ndf = ndf.append({'Event':event,'Driver':driver,'Trip':trip,'Starttime':starttime,'Endtime':endtime,'Speedingtime':duration}, ignore_index=True)
        #write row
        event += 1
        driver = row['Driver']
        trip = row['Trip']
        time = row['Time']
        starttime = time
        endtime = time
        duration = 10
ndf.to_csv('result.csv',index=False,float_format='%.0f')