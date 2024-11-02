import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_csv(r"C:\Users\murat\Desktop\YPZ\veriler.csv")
print(veriler) 

x = veriler.iloc[:,1:4].values #bağımsız değişkenler
y = veriler.iloc[:,4:].values #bağımlı değişken
print(y)

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test) 


from sklearn.svm import SVC
svc = SVC(kernel="linear")
svc.fit(x_train,y_train)
y_pred = svc.predict(x_test)
print(y_pred)

svc = SVC(kernel="rbf")
svc.fit(x_train,y_train)
y_pred = svc.predict(x_test)
print(y_pred)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred,y_test) 
print("SVC") 
print(cm) 

