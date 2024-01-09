import pandas as pd



df = pd.read_csv('Data/TeenTailgating.csv')
df = df.sort_values(by=['Driver', 'Trip', 'Time'])
ndf = pd.DataFrame(columns=['Event','Driver','Trip','Starttime','Endtime'])

event = 1
for index,row in df.iterrows():
    print('Start!')
    driver = row['Driver']
    trip = row['Trip']
    time = row['Time']
    starttime = time
    endtime = time
    break

for index,row in df.iterrows():
    if row['Driver'] == driver and row['Trip'] == trip and row['Time'] == time:
        continue
    elif row['Driver'] == driver and row['Trip'] == trip and abs(row['Time'] - time) <= 1000:
        time = row['Time']
        endtime = time
    else:
        if driver != row['Driver']:
            print(driver)
        ndf = ndf.append({'Event':event,'Driver':driver,'Trip':trip,'Starttime':starttime,'Endtime':endtime}, ignore_index=True)
        #write row
        event += 1
        driver = row['Driver']
        trip = row['Trip']
        time = row['Time']
        starttime = time
        endtime = time
ndf.to_csv('result.csv',index=False,float_format='%.0f')