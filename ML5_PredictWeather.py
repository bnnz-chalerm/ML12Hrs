#https://kongruksiamza.medium.com/%E0%B8%AA%E0%B8%A3%E0%B8%B8%E0%B8%9B-machine-learning-ep-3-%E0%B8%81%E0%B8%B2%E0%B8%A3%E0%B8%A7%E0%B8%B4%E0%B9%80%E0%B8%84%E0%B8%A3%E0%B8%B2%E0%B8%B0%E0%B8%AB%E0%B9%8C%E0%B8%81%E0%B8%B2%E0%B8%A3%E0%B8%96%E0%B8%94%E0%B8%96%E0%B8%AD%E0%B8%A2%E0%B9%80%E0%B8%8A%E0%B8%B4%E0%B8%87%E0%B9%80%E0%B8%AA%E0%B9%89%E0%B8%99-linear-regression-891260e4a957
import os
from turtle import color
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

project_path = os.getcwd()
print(project_path)
data_path = project_path + "\ML12Hrs\Data\Weather.csv"
dataset = pd.read_csv(data_path)

# train & test set
x = dataset["MinTemp"].values.reshape(-1,1)
y = dataset["MaxTemp"].values.reshape(-1,1)

#80% - 20%
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#training
model = LinearRegression()
model.fit(x_train,y_train)

#test
#ทดสอบโมเดล จะได้ค่าที่ทำนายจากนั้นค่อยเช็คกับตัว y_test ที่เป็นข้อมูลจริง
y_pred = model.predict(x_test) 

#หาค่าความคลาดเคลื่อนระหว่างค่าที่ได้จากทำงาน และค่าจริงๆ
print("MAE = ",metrics.mean_absolute_error(y_test,y_pred))
print("MSE = ",metrics.mean_squared_error(y_test,y_pred))
print("RMSE = ",np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
#แสดงค่าความแม่นยำด้วย R-Square หากมีค่าเป็น 1 แสดงว่าแม่นยำที่สุด
print("Score = ",metrics.r2_score(y_test,y_pred))


# compare true data & predict data
#.flatten() แปลงอาเรย์สองมิติเป็น 1 มิติ
df=pd.DataFrame({'Actually':y_test.flatten(),'Predicted':y_pred.flatten()})
df1 = df.head(20)
df1.plot(kind="bar",figsize=(16,10))
plt.show()



# print(df.head())

# plt.scatter(x_test,y_test)
# plt.plot(x_test,y_pred,color="red",linewidth=2)
# plt.show()



#print(dataset.describe())

# dataset.plot(x='MinTemp',y='MaxTemp',style='o')
# plt.title("min & max temp")
# plt.xlabel("min temp")
# plt.ylabel("max temp")
# plt.show()



