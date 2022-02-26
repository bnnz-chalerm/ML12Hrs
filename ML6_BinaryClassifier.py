#https://kongruksiamza.medium.com/%E0%B8%AA%E0%B8%A3%E0%B8%B8%E0%B8%9B-machine-learning-ep-4-%E0%B8%95%E0%B8%B1%E0%B8%A7%E0%B8%88%E0%B8%B3%E0%B9%81%E0%B8%99%E0%B8%81%E0%B9%81%E0%B8%9A%E0%B8%9A%E0%B9%84%E0%B8%9A%E0%B8%A3%E0%B8%B2%E0%B8%A3%E0%B8%B5%E0%B9%88-binary-classifier-6ebc8e1a5e61
#SGD Stochastic Gradient Descent
import os
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier

def displayImage(x):
    plt.imshow(x.reshape(28,28),
    cmap=plt.cm.binary,
    interpolation="nearest")
    plt.show()

def displayPredict(clf,actually_y,position):
    print("actually = ",actually_y)
    print("prediction = ")

project_path = os.getcwd()

data_path = project_path + "\ML12Hrs\Data\mnist-original.mat"

mnist_raw = loadmat(data_path)
mnist={
    "data":mnist_raw["data"].T,
    "target":mnist_raw["label"][0]
}
x,y = mnist["data"],mnist["target"]
# print(mnist["data"].shape)
# print(mnist["target"].shape)

# train & test set
#1-60000
#60001-70000
x_train,x_test,y_train,y_test = x[:60000],x[60000:],y[:60000],y[60000:]
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# class 0 , non class 0
#จะทำการทดสอบ ว่าในตำแหน่งที่ 5000 เป็นเลข 0 หรือไม่ model จะตอบ true/false
#y_train =[0,0,0,....,9,9,9,...,9]
#y_train_0 = (y_train==0) จะทำให้ตัวเลขเป็น bool => [true,true,true,...,false,false,...,false]
predict_no = 5000
y_train_0 = (y_train==0)
y_test_0 = (y_test==0)

print(y_train_0.shape,y_train_0)
print(y_test_0.shape,y_test_0)

#SGD model
sgd_slf = SGDClassifier()
#อยากให้โมเดลเทรนแค่ว่าเป็น 0 หรือไม่ใช่ จึงใช้ y_train_0 ค่า classifier จะได้ 2 รูปแบบ
sgd_slf.fit(x_train,y_train_0)

displayImage(x_test[predict_no])

