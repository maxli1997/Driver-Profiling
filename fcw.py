import pandas as pd
from matplotlib import pyplot as plt

# separate teen and adult drivers
driver_df = pd.read_csv('Setup_2/BGM_all_3.csv',usecols=['Driver','Risk Level'])
c_teen = []
c_adult = []
r_teen = []
r_adult = []
for index,row in driver_df.iterrows():
    driver = row['Driver']
    if driver < 100:
        if row['Risk Level'] == 0:
            c_teen.append(driver)
        else:
            r_teen.append(driver)
    else:
        if row['Risk Level'] == 0:
            c_adult.append(driver/100)
        else:
            r_adult.append(driver/100)

# calculate valid trips
trip_df_1 = pd.read_csv('Data/KMeans/TeenTripWeek.csv',usecols=['Driver','Trip','Week'])
trip_df_2 = pd.read_csv('Data/Lv_KMeans/LvTripWeek.csv',usecols=['Driver','Trip','Week'])
teen_trips = {}
adult_trips = {}
for driver in c_teen+r_teen:
    temp = trip_df_1[trip_df_1['Driver']==driver]
    temp = temp[(temp['Week']>=1) & (temp['Week']<=2)]
    trips = pd.Series.to_list(temp['Trip'])
    teen_trips[driver] = trips
for driver in c_adult+r_adult:
    temp = trip_df_2[trip_df_2['Driver']==driver]
    temp = temp[(temp['Week']>=1) & (temp['Week']<=2)]
    trips = pd.Series.to_list(temp['Trip'])
    adult_trips[driver] = trips

dr = []
tp = []
st= []
level = []
age = []
# calculate brake time
data_df_1 = pd.read_csv('Data/RawData/Teen_Fcw_Data.csv',usecols=['Driver','Trip','Time','Brake','StartTime','EndTime','Ax'])
data_df_2 = pd.read_csv('Data/RawData/Adult_Fcw_Data.csv',usecols=['Driver','Trip','Time','Brake','StartTime','EndTime','Ax'])
c_brake_time = []
c_total_time = []
c_min_ax = []
for driver in c_teen:
    trips = teen_trips[driver]
    for trip in trips:
        temp = data_df_1[(data_df_1['Driver']==driver) & (data_df_1['Trip']==trip)]
        if not temp.empty:
            event_id = temp.iloc[0]['StartTime']
            min_ax = temp.iloc[0]['Ax']
            time = 500
            total_time = 0
            b_flag = 0
            for index,row in temp.iterrows():
                current_id = row['StartTime']
                if event_id != current_id:
                    dr.append(driver)
                    tp.append(trip)
                    st.append(event_id)
                    level.append(0)
                    age.append(0)
                    c_brake_time.append(time)
                    c_total_time.append(total_time)
                    c_min_ax.append(min_ax)

                    event_id = current_id
                    time = 500
                    total_time = 0
                    min_ax = row['Ax']
                    b_flag = 0
                    continue
                if row['Brake'] == 1 and b_flag == 0:
                    time = row['Time'] - row['StartTime']
                    starttime = row['Time']
                    b_flag = 1
                elif row['Brake'] == 1 and b_flag == 1:
                    total_time = row['Time']-starttime
                if row['Ax'] < min_ax:
                    min_ax = row['Ax']
            dr.append(driver)
            tp.append(trip)
            st.append(event_id)
            level.append(0)
            age.append(0)
            c_brake_time.append(time)
            c_total_time.append(total_time)
            c_min_ax.append(min_ax)
                    
for driver in c_adult:
    trips = adult_trips[driver]
    for trip in trips:
        temp = data_df_2[(data_df_2['Driver']==driver) & (data_df_2['Trip']==trip)]
        if not temp.empty:
            event_id = temp.iloc[0]['StartTime']
            min_ax = temp.iloc[0]['Ax']
            time = 500
            total_time = 0
            b_flag = 0
            for index,row in temp.iterrows():
                current_id = row['StartTime']
                if event_id != current_id:
                    dr.append(driver)
                    tp.append(trip)
                    st.append(event_id)
                    level.append(0)
                    age.append(1)
                    c_brake_time.append(time)
                    c_total_time.append(total_time)
                    c_min_ax.append(min_ax)

                    event_id = current_id
                    time = 500
                    min_ax = row['Ax']
                    total_time = 0
                    b_flag = 0
                    continue
                if row['Brake'] == 1 and b_flag == 0:
                    time = row['Time'] - row['StartTime']
                    starttime = row['Time']
                    b_flag = 1
                elif row['Brake'] == 1 and b_flag == 1:
                    total_time = row['Time']-starttime
                if row['Ax'] < min_ax:
                    min_ax = row['Ax']   
            dr.append(driver)
            tp.append(trip)
            st.append(event_id)
            level.append(0)
            age.append(1)
            c_brake_time.append(time)
            c_total_time.append(total_time)
            c_min_ax.append(min_ax)      
r_brake_time = []
r_total_time = []
r_min_ax = []
for driver in r_teen:
    trips = teen_trips[driver]
    for trip in trips:
        temp = data_df_1[(data_df_1['Driver']==driver) & (data_df_1['Trip']==trip)]
        if not temp.empty:
            event_id = temp.iloc[0]['StartTime']
            min_ax = temp.iloc[0]['Ax']
            time = 500
            total_time = 0
            b_flag = 0
            for index,row in temp.iterrows():
                current_id = row['StartTime']
                if event_id != current_id:
                    dr.append(driver)
                    tp.append(trip)
                    st.append(event_id)
                    level.append(1)
                    age.append(0)
                    r_brake_time.append(time)
                    r_total_time.append(total_time)
                    r_min_ax.append(min_ax)

                    event_id = current_id
                    time = 500
                    min_ax = row['Ax']
                    total_time = 0
                    b_flag = 0
                    continue
                if row['Brake'] == 1 and b_flag == 0:
                    time = row['Time'] - row['StartTime']
                    starttime = row['Time']
                    b_flag = 1
                elif row['Brake'] == 1 and b_flag == 1:
                    total_time = row['Time']-starttime
                if row['Ax'] < min_ax:
                    min_ax = row['Ax']
            dr.append(driver)
            tp.append(trip)
            st.append(event_id)
            level.append(1)
            age.append(0)
            r_brake_time.append(time)
            r_total_time.append(total_time)
            r_min_ax.append(min_ax)
for driver in r_adult:
    trips = adult_trips[driver]
    for trip in trips:
        temp = data_df_2[(data_df_2['Driver']==driver) & (data_df_2['Trip']==trip)]
        if not temp.empty:
            event_id = temp.iloc[0]['StartTime']
            min_ax = temp.iloc[0]['Ax']
            time = 500
            total_time = 0
            b_flag = 0
            for index,row in temp.iterrows():
                current_id = row['StartTime']
                if event_id != current_id:
                    dr.append(driver)
                    tp.append(trip)
                    st.append(event_id)
                    level.append(1)
                    age.append(1)
                    r_brake_time.append(time)
                    r_total_time.append(total_time)
                    r_min_ax.append(min_ax)

                    event_id = current_id
                    time = 500
                    min_ax = row['Ax']
                    total_time = 0
                    b_flag = 0
                    continue
                if row['Brake'] == 1 and b_flag == 0:
                    time = row['Time'] - row['StartTime']
                    starttime = row['Time']
                    b_flag = 1
                elif row['Brake'] == 1 and b_flag == 1:
                    total_time = row['Time']-starttime
                if row['Ax'] < min_ax:
                    min_ax = row['Ax']
            dr.append(driver)
            tp.append(trip)
            st.append(event_id)
            level.append(1)
            age.append(1)
            r_brake_time.append(time)
            r_total_time.append(total_time)
            r_min_ax.append(min_ax)

# save results
result_list = [dr,tp,st,level,age,c_brake_time+r_brake_time,c_total_time+r_total_time,c_min_ax+r_min_ax]
result = pd.DataFrame(result_list).transpose()
result.columns = ['Driver','Trip','StartTime','Risky Level','Age','Brake Reaction Time','Brake Duration','Max Deceleration']
result.to_csv('Setup_2/fcw_stats.csv',index=False)