import os
from turtle import color
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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



