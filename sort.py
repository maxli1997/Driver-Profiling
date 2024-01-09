import pandas as pd

iter_csv = pd.read_csv('Data/TeenTailgatingData(Unsorted).csv',iterator=True,chunksize=1000000)
df = pd.concat([chunk for chunk in iter_csv])
df = df.sort_values(by=['Event','Driver','Trip','Time'])

df.to_csv('result.csv',index=False)
