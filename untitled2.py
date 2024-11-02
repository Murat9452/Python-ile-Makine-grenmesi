import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
veriler = pd.read_csv(r"C:\Users\murat\Desktop\YPZ\veriler.csv")
print(veriler)
veriler.info()
boy = veriler[["boy"]]
print(boy)
boykilo = veriler[["boy","kilo"]]
print(boykilo)


from sklearn.impute import SimpleImputer 
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
yas = veriler.iloc[:,1:4].values
print(yas)
imputer = imputer.fit(yas[:,1:4])
yas[:,1:4] = imputer.transform(yas[:,1:4])
print(yas) 


ülke = veriler.iloc[:,0:1].values
print(ülke)
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
ülke[:,0] = le.fit_transform(veriler.iloc[:,0])
print(ülke)
ohe = preprocessing.OneHotEncoder()
ülke = ohe.fit_transform(ülke).toarray()
print(ülke)

c = veriler.iloc[:,-1:].values
print(c)
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
c[:,-1] = le.fit_transform(veriler.iloc[:,-1])
print(c)
ohe = preprocessing.OneHotEncoder()
c = ohe.fit_transform(c).toarray()
print(c)


sonuc = pd.DataFrame(data=ülke, index= range(22), columns=["fr","tr","us"])
print(sonuc) 
sonuc2 = pd.DataFrame(data=yas, index= range(22), columns=["boy","kilo","yas"])
print(sonuc2)
cinsiyet = veriler.iloc[:,-1].values 
print(cinsiyet)
sonuc3 = pd.DataFrame(data= c[:,:1], index= range(22),columns=["cinsiyet"])
print(sonuc3)

s = pd.concat([sonuc,sonuc2], axis=1)
print(s)
s2 = pd.concat([s,sonuc3], axis=1)
print(s2) 

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(s, sonuc3, test_size=0.33, random_state=0)

from sklearn.linear_model import LinearRegression
regrossor = LinearRegression()
regrossor.fit( x_train, y_train)

y_pred = regrossor.predict(x_test)

boy = s2.iloc[:,3:4].values
print(boy)

sol = s2.iloc[:,:3]
sağ = s2.iloc[:,:4]

veri = pd.concat([sol,sağ],axis=1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(veri, boy, test_size=0.33, random_state=0)


from sklearn.linear_model import LinearRegression
r2 = LinearRegression()
r2.fit(x_train, y_train)

y_pred = r2.predict(x_test)

import statsmodels.api as sm

X = np.append(arr= np.ones((22,1)).astype(int), values=veri, axis=1)

X_l = veri.iloc[:,[0,1,2,3,4,5]].values
X_l = np.array(X_l, dtype=float)
model = sm.OLS(boy,X_l).fit()
print(model.summary())
 

