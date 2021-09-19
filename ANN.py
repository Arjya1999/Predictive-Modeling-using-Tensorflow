import pandas as pd
import numpy as np

dataset=pd.read_csv("datasetA.csv")

dataset.describe()

null_data=dataset.isnull().sum()

name_lr=dataset['lr'].value_counts().index
name_lr = name_lr[:-1,]
data_lr =dataset['lr'].value_counts()

for i in name_lr:
    dataset[i+"_lr"]=np.where(dataset["lr"]==i,1,0)
    
dataset.drop("lr",inplace=True,axis=1)

dataset["lc"]=pd.get_dummies(dataset["lc"],drop_first=True)

name_cr=dataset['cr'].value_counts().index
name_cr=name_cr[:-1,]
data_cr =dataset['cr'].value_counts()

for i in name_cr:
    dataset[i+"_cr"]=np.where(dataset["cr"]==i,1,0)
dataset.drop("cr",inplace=True,axis=1)

name_rg=dataset['rg'].value_counts().index
name_rg=name_rg[:-1,]
data_rg =dataset['rg'].value_counts()

for i in name_rg:
    dataset[i+"_rg"]=np.where(dataset["rg"]==i,1,0)  
dataset.drop("rg",inplace=True,axis=1)

name_tg = dataset['tg'].value_counts().index
name_tg = name_tg[:-1,]
data_tg = dataset['tg'].value_counts()

for i in name_tg:
    dataset[i+"_tg"]=np.where(dataset["tg"]==i,1,0)
dataset.drop("tg",inplace=True,axis=1)

name_rc= dataset['rc'].value_counts().index
name_rc= name_rc[:-1,]
data_rc= dataset['rc'].value_counts()

for i in name_rc:
    dataset[i+"_rc"]=np.where(dataset["rc"]==i,1,0)
dataset.drop("rc",inplace=True,axis=1)

name_tc=dataset['tc'].value_counts().index
name_tc=name_tc[:-1,]
data_tc =dataset['tc'].value_counts()

for i in name_tc:
    dataset[i+"_tc"]=np.where(dataset["tc"]==i,1,0)
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

data_dc = dataset['dc'].value_counts()
name_dc =dataset['dc'].value_counts().index
name_dc =name_dc[:-1,]

for i in name_dc:
    dataset[i+"_dc"]=np.where(dataset["dc"]==i,1,0)
dataset.drop("dc",inplace=True,axis=1)

data_predict = dataset["predictA"].value_counts()

dataset["predictA"]=np.where(dataset["predictA"]=="TNP",None,dataset["predictA"])
dataset["predictA"]=np.where(dataset["predictA"]=="PU",None,dataset["predictA"])

null_data=dataset.isnull().sum()
dataset.dropna(inplace=True)

data_predict = dataset["predictA"].value_counts()

dataset.reset_index(inplace=True)

dataset.drop("py",inplace=True,axis=1)
dataset.drop("index",inplace=True,axis=1)

data_predict = dataset["predictA"].value_counts()
name = dataset['predictA'].value_counts().index

dataset["1"]=np.where(dataset["predictA"]=="1",1,0)
dataset["2"]=np.where(dataset["predictA"]=="2",1,0)
dataset["3"]=np.where(dataset["predictA"]=="3",1,0)
dataset["4"]=np.where(dataset["predictA"]=="4",1,0)
dataset["5"]=np.where(dataset["predictA"]=="5",1,0)
dataset["6"]=np.where(dataset["predictA"]=="6",1,0)
dataset["7"]=np.where(dataset["predictA"]=="7",1,0)
dataset["8"]=np.where(dataset["predictA"]=="8",1,0)
dataset["9"]=np.where(dataset["predictA"]=="9",1,0)
dataset["10"]=np.where(dataset["predictA"]=="10",1,0)
dataset["11"]=np.where(dataset["predictA"]=="11",1,0)
dataset["12"]=np.where(dataset["predictA"]=="12",1,0)
dataset["13"]=np.where(dataset["predictA"]=="13",1,0)
dataset["14"]=np.where(dataset["predictA"]=="14",1,0)

dataset.drop("predictA",inplace=True,axis=1)

X = dataset.iloc[:,:-14].values
y = dataset.iloc[:,-14:].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Importing the Keras libraries and packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 200, kernel_initializer = 'uniform', activation = 'relu', input_dim = 134))
#classifier.add(Dropout(rate = 0.2))

# Adding the second hidden layer
classifier.add(Dense(units = 200, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(rate = 0.2))

# Adding the third hidden layer
classifier.add(Dense(units = 200, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(rate = 0.2))

# Adding the output layer
classifier.add(Dense(units = 14, kernel_initializer = 'uniform', activation = 'softmax'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
history = classifier.fit(X_train, y_train, validation_data=(X_test,y_test),batch_size = 1000, epochs = 50)

import matplotlib.pyplot as plt

loss_train = history.history['loss']
loss_val = history.history['val_loss']
epochs = range(50)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_test = np.argmax(y_test,axis=1)
y_pred=np.argmax(y_pred,axis=1)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)

test_dataset = pd.read_csv("datasetA_predict.csv")
predicted_dataset=pd.read_csv("datasetA_predict.csv")

for i in name_lr:
    test_dataset[i+"_lr"]=np.where(test_dataset["lr"]==i,1,0)  
test_dataset.drop("lr",inplace=True,axis=1)

test_dataset["lc"]=np.where(test_dataset["lc"]=="A",0,1)

for i in name_cr:
    test_dataset[i+"_cr"]=np.where(test_dataset["cr"]==i,1,0)
test_dataset.drop("cr",inplace=True,axis=1)

for i in name_rg:
    test_dataset[i+"_rg"]=np.where(test_dataset["rg"]==i,1,0)  
test_dataset.drop("rg",inplace=True,axis=1)

for i in name_tg:
    test_dataset[i+"_tg"]=np.where(test_dataset["tg"]==i,1,0)
test_dataset.drop("tg",inplace=True,axis=1)

for i in name_rc:
    test_dataset[i+"_rc"]=np.where(test_dataset["rc"]==i,1,0)
test_dataset.drop("rc",inplace=True,axis=1)

for i in name_tc:
    test_dataset[i+"_tc"]=np.where(test_dataset["tc"]==i,1,0)
test_dataset.drop("tc",inplace=True,axis=1)

test_dataset["w1t"]=np.where(test_dataset["w1t"]=="False",0,1)
test_dataset["w2t"]=np.where(test_dataset["w2t"]=="False",0,1)
test_dataset["w3t"]=np.where(test_dataset["w3t"]=="False",0,1)
test_dataset["w4t"]=np.where(test_dataset["w4t"]=="False",0,1)
test_dataset["w5t"]=np.where(test_dataset["w5t"]=="False",0,1)

for i in name_dc:
    test_dataset[i+"_dc"]=np.where(test_dataset["dc"]==i,1,0)
    
test_dataset.drop("dc",inplace=True,axis=1)
test_dataset.drop("py",inplace=True,axis=1)

test=test_dataset.iloc[:,:].values

test = sc.transform(test)
predictions = classifier.predict(test)

predictions = np.argmax(predictions,axis=1)
predictions = predictions+1

predicted_dataset["predictA"]=predictions

predicted_dataset.to_csv("predicted_dataset.csv")