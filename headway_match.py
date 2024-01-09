import pandas as pd



df = pd.read_csv('Data/FullData/TeenTailgatingHeadway.csv')
ndf = pd.DataFrame(columns=['Event','Driver','Trip','Starttime','Endtime'])

events = df.Event.unique()

for event in events:
    temp_df = df[df['Event']==event]
    temp_df = temp_df.sort_values(by=['Time'])
    start_flag = 0
    end_flag = 0
    start = 0
    end = 0
    for index,row in temp_df.iterrows():
        headway = row['headway']
        if start_flag == 0 and headway <= 1:
            start = row['Time']
            start_flag = 1
        elif start_flag == 1 and end_flag == 1 and headway <= 1:
            end_flag = 0
        if start_flag == 1 and end_flag == 0 and headway > 1:
            end = row['Time']-10
            end_flag = 1
    if end_flag == 0:
        end = row['Time']-10
    ndf = ndf.append({'Event':event,'Starttime':start,'Endtime':end,'Driver':row['Driver'],'Trip':row['Trip']}, ignore_index=True)


ndf.to_csv('result.csv',index=False,float_format='%.0f')