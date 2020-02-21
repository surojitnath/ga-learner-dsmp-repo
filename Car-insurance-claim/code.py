# --------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Code starts here

df = pd.read_csv(path)
df.head()
print(df.info())
cat_col =['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']

for i in cat_col:
    df[i] = df[i].str.replace(r'\D', '')
X= df.iloc[:,0:-1]
y= df.iloc[:,-1]   
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=6) 
# Code ends here


# --------------
# Code starts here

for i in cat_col:
    X_train[i] = X_train[i].astype('float')
    X_test[i] = X_test[i].astype('float')

print("null values of train",X_train.isnull().sum())
print("null values of test",X_test.isnull().sum())

# Code ends here


# --------------
# Code starts here
X_train.dropna(subset = ['YOJ','OCCUPATION'],inplace =True)
X_test.dropna(subset = ['YOJ','OCCUPATION'],inplace =True)
y_train=y_train[X_train.index]
y_test=y_test[X_test.index]
missing_val=['AGE','CAR_AGE','INCOME','HOME_VAL']
for i in missing_val:
    X_train[i].fillna((X_train[i].mean()), inplace=True)
    X_test[i].fillna((X_test[i].mean()), inplace=True)
# Code ends here


# --------------
from sklearn.preprocessing import LabelEncoder
columns = ["PARENT1","MSTATUS","GENDER","EDUCATION","OCCUPATION","CAR_USE","CAR_TYPE","RED_CAR","REVOKED"]

# Code starts here

label_encoder = LabelEncoder() 
  
# Encode labels in column 
for k in columns:
    X_train[k]= label_encoder.fit_transform(X_train[k]) 
    X_test[k]= label_encoder.fit_transform(X_test[k]) 
# Code ends here



# --------------
from sklearn.metrics import precision_score 
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression



# code starts here 

model =LogisticRegression(random_state=0)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
score = accuracy_score(y_test,y_pred)
print("Score =",score)

# Code ends here


# --------------
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# code starts here
smote = SMOTE(random_state =9)
X_train , y_train =smote.fit_sample(X_train,y_train)
scaler=StandardScaler()
scaler.fit_transform(X_train,y_train)
X_test=scaler.transform(X_test)

# Code ends here


# --------------
# Code Starts here

model =LogisticRegression(random_state=0)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
score = accuracy_score(y_test,y_pred)
print("Score :",score)
# Code ends here


