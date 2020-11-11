import pandas as pd
filename="./data/train.csv"
df=pd.read_csv(filename)
df=df[["Text","Label"]]
drop_index=df.index[df["Label"]=="label"]
df=df.drop(drop_index)
df.to_csv(filename,index=False)
