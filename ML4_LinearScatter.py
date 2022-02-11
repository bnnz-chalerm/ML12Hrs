#y = ax + b
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

rng= np.random

#การจำลองข้อมูล
x = rng.rand(50)*10
y = 2*x+rng.randn(50)
#print(x)

#linear regression model
model = LinearRegression()

#แปลงอาเรย์ 1 มิติเป็น 2 มิติ
x_new = x.reshape(-1,1)
#print(x_new)

#train algorithm
model.fit(x_new,y)

#test model
xfit = np.linspace(-1,11)
xfit_new = xfit.reshape(-1,1)
#print(xfit_new.shape)
yfit = model.predict(xfit_new)


#analysis model
# #Intercept, Coefficient, R-SQuare
# print(model.score(x_new,y))
# print(model.intercept_)
# print(model.coef_)
plt.scatter(x,y)
plt.plot(xfit,yfit)
plt.show()



# print(y)

# plt.scatter(x,y)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.grid()
# plt.show()

