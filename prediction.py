import pandas as pd

dataset=pd.read_csv("datasetA.csv")

dataset.describe()

null_data=dataset.isnull().sum()

dataset.drop(["nameA","nameB","w1btd1","w2btd1","w3btd1","w4btd1","w6btd1","mtv","morn","dc","l2dc","l3dc","IWin"],inplace=True,axis=1)
null_data=dataset.isnull().sum()

p1=dataset['p1'].value_counts()

dataset['p1'].fillna()