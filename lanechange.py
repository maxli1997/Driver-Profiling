import pandas as pd

out_df = pd.DataFrame(columns=['Event','Driver','Trip','Starttime','Endtime','LaneChangeStarttime','LaneChangeEndtime','DerivedEndtime'])
# read pre-processed data
event_df = pd.read_csv('Data/TeenLC.csv')
iter_csv = pd.read_csv('Data/TeenRaw.csv',iterator=True,chunksize=1000000)
in_df = pd.concat([chunk for chunk in iter_csv])

event_id = 1
# iterate through every event
for index,event in event_df.iterrows():
    print(index)
    #test run
    #if event_id == 10:
    #    break

    driver = event['Driver']
    trip = event['Trip']
    start = event['StartTime']
    end = event['EndTime']

    # load data with same driver and trip
    raw_df = in_df[(in_df['Driver']==driver) & (in_df['Trip']==trip)]

    flag = 0
    for time in range(start,end+10,10):
        if (raw_df['Time']==time).any() == False:
            flag = 1
            break
    if flag == 1:
        continue

    # check that previous 15s on highway loop through all 150 entries
    flag = 0
    for offset in range(10,1510,10):
        if (raw_df['Time']==start-offset).any() == False:
            flag = 1
            break
    if flag == 1:
        continue
    startpoint = start-1500

    # find the end point
    offset = 0
    current_l = raw_df.loc[raw_df['Time']==end]['LaneOffset'].tolist()
    current_l = current_l[0]
    if current_l > 0:
        e_type = 1
    else:
        e_type = 2
    endpoint = -1
    while True:
        offset += 10
        if (raw_df['Time']==offset+end).any() == False:
            break
        next_l = raw_df[raw_df['Time']==offset+end]['LaneOffset'].tolist()
        next_l = next_l[0]
        # check end type B
        if next_l == 0:
            endpoint = offset+end
            break
        elif current_l * next_l < 0:
            endpoint = offset+end
            break
        # check end type C?
        if (e_type == 1) & (next_l-current_l >= 0.02):
            endpoint = offset+end
            break
        elif (e_type == 2) & (next_l-current_l <= -0.02):
            endpoint = offset+end
            break
        current_l = next_l
    if endpoint == -1:
        continue

    # check for 50s after endpoint
    flag = 0
    for offset in range(10,5010,10):
        if (raw_df['Time']==endpoint+offset).any() == False:
            flag = 1
            break
    if flag == 1:
        continue
    derived = endpoint+300

    # pass all tests report event
    out_df = out_df.append({'Event':event_id,'Driver':driver,'Trip':trip,'Starttime':startpoint,'Endtime':derived,'LaneChangeStarttime':start,'LaneChangeEndtime':end,'DerivedEndtime':endpoint}, ignore_index=True)
    event_id += 1

out_df.to_csv('result.csv',index=False,float_format='%.0f')