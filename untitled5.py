#1.kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# veri yükleme
veriler = pd.read_csv(r"C:\Users\murat\Desktop\YPZ\maaslar.csv")
print(veriler) 


# data frame dilimleme(slice)
x= veriler.iloc[:,1:2]
y= veriler.iloc[:,2:]

# Numpy dizi (array) dönüşümü
X = x.values
Y = y.values

# Linear Regression
# Doğrusal model oluşturma
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, Y)

# Görselleştirme
plt.scatter(X, Y,color="red")
plt.plot(x, lin_reg.predict(X),color="blue") 
plt.show()


# Polynomal Regression
# Doğrusal olmayan (nonlinear model oluşturma)
# 2. dereceden polinom
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)

# 4. dereceden polinom
from sklearn.preprocessing import PolynomialFeatures
poly_reg3 = PolynomialFeatures(degree = 4)
x_poly3 = poly_reg3.fit_transform(X)
print(x_poly3)
lin_reg3 = LinearRegression()
lin_reg3.fit(x_poly3,y)


plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg3.predict(poly_reg3.fit_transform(X)), color = 'blue')
plt.show() 


print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))

print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))
print(lin_reg2.predict(poly_reg.fit_transform([[11]]))) 


# VERİLERİN ÖLÇEKLLENMESİ
from sklearn.preprocessing import StandardScaler
sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)
sc2 = StandardScaler()
y_olcekli = np.ravel(sc2.fit_transform(Y.reshape(-1,1)))

# SVR
from sklearn.svm import SVR
svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_olcekli,y_olcekli)
plt.scatter(x_olcekli,y_olcekli,color='red')
plt.plot(x_olcekli,svr_reg.predict(x_olcekli),color='blue')
plt.show()
print(svr_reg.predict([[11]]))
print(svr_reg.predict([[6.6]]))

# DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)
Z = X + 0.5
K = X - 0.4
plt.scatter(X,Y, color='red')
plt.plot(x,r_dt.predict(X), color='blue')
plt.plot(x,r_dt.predict(Z),color='green')
plt.plot(x,r_dt.predict(K),color='yellow')
plt.show()
print(r_dt.predict([[11]]))
print(r_dt.predict([[6.6]])) 


# RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=10,random_state=0)
rf_reg.fit(X,Y.ravel())
print(rf_reg.predict([[6.6]]))
print(rf_reg.predict([[11]])) 
plt.scatter(X,Y, color="red")
plt.plot(X,rf_reg.predict(X), color="blue")
plt.plot(X,rf_reg.predict(Z), color="green")
plt.plot(X,rf_reg.predict(K),color="yellow")
plt.show() 





















