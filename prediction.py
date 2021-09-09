import pandas as pd
import numpy as np

dataset=pd.read_csv("datasetA.csv")

dataset.describe()

null_data=dataset.isnull().sum()

name=dataset['lr'].value_counts().index
data_lr =dataset['lr'].value_counts()

for i in name:
    dataset[i+"_lr"]=np.where(dataset["lr"]==i,1,0)
dataset.drop("L2_lr",inplace=True,axis=1)
dataset.drop("lr",inplace=True,axis=1)

dataset["lc"]=pd.get_dummies(dataset["lc"],drop_first=True)

name=dataset['cr'].value_counts().index
data_cr =dataset['cr'].value_counts()

for i in name:
    dataset[i+"_cr"]=np.where(dataset["cr"]==i,1,0)
dataset.drop("B+2_cr",inplace=True,axis=1)
dataset.drop("cr",inplace=True,axis=1)

name=dataset['rg'].value_counts().index
data_rg =dataset['rg'].value_counts()

for i in name:
    dataset[i+"_rg"]=np.where(dataset["rg"]==i,1,0)  
dataset.drop("f2_rg",inplace=True,axis=1)
dataset.drop("rg",inplace=True,axis=1)

name=dataset['tg'].value_counts().index
data_tg =dataset['tg'].value_counts()

for i in name:
    dataset[i+"_tg"]=np.where(dataset["tg"]==i,1,0)
dataset.drop("g7_tg",inplace=True,axis=1)
dataset.drop("tg",inplace=True,axis=1)

name=dataset['rc'].value_counts().index
data_rc =dataset['rc'].value_counts()

for i in name:
    dataset[i+"_rc"]=np.where(dataset["rc"]==i,1,0)
dataset.drop("r49_rc",inplace=True,axis=1)
dataset.drop("rc",inplace=True,axis=1)

name=dataset['tc'].value_counts().index
data_tc =dataset['tc'].value_counts()

for i in name:
    dataset[i+"_tc"]=np.where(dataset["tc"]==i,1,0)
dataset.drop("t23_tc",inplace=True,axis=1)
dataset.drop("tc",inplace=True,axis=1)

data_w1t=dataset['w1t'].value_counts()
dataset["w1t"]=pd.get_dummies(dataset["w1t"],drop_first=True)


data_w2t=dataset['w2t'].value_counts()
dataset["w2t"]=pd.get_dummies(dataset["w2t"],drop_first=True)

data_w3t=dataset['w3t'].value_counts()
data_w4t=dataset['w4t'].value_counts()
data_w5t=dataset['w5t'].value_counts()

dataset["w3t"]=pd.get_dummies(dataset["w3t"],drop_first=True)
dataset["w4t"]=pd.get_dummies(dataset["w4t"],drop_first=True)
dataset["w5t"]=pd.get_dummies(dataset["w5t"],drop_first=True)

data_dc1 = dataset['dc1'].value_counts()
name=dataset['dc1'].value_counts().index

for i in name:
    dataset[i+"_dc1"]=np.where(dataset["dc1"]==i,1,0)
dataset.drop("zc_dc1",inplace=True,axis=1)
dataset.drop("dc1",inplace=True,axis=1)

data_predict = dataset["predictA"].value_counts()

check=dataset["predictA"]=="TNP"

dataset.drop(dataset["predictA"]=="TNP",inplace=True,axis=0)