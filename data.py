import numpy as np
import pandas as pd
filename="./data/train_dumy.csv"
df=pd.read_csv(filename)
drop_index=[]
df=df.dropna(how="any")
label=df.at[1,"Label"]
if type(label) is np.float64:
    for index, row in df.iterrows():
        label=df.at[index,"Label"]
        if type(label) is not np.float64 or int(label)<0 or int(label) >=20:
            drop_index.append(index)
if type(label) is np.int64:
    for index, row in df.iterrows():
        label=df.at[index,"Label"]
        if type(label) is not np.int64 or int(label)<0 or int(label) >=20:
            drop_index.append(index)

if type(label) is int:
    for index, row in df.iterrows():
        label=df.at[index,"Label"]
        if type(label) is not int or int(label)<0 or int(label) >=20:
            drop_index.append(index)

df=df.drop(drop_index)
df["Label"]=df["Label"].astype(int)
df.to_csv(filename,index=False)
