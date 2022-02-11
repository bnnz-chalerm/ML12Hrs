from scipy.io import loadmat
import matplotlib.pyplot as plt 
import os

project_path = os.getcwd()
data_path = "\ML\Data\mnist-original.mat"
#print(os.getcwd() )
#mnist_raw = loadmat("C:\Research\PythonProject\Basic_ML\ML\data\mnist-original.mat")
mnist_raw = loadmat(os.path.join(project_path,"ML/Data","mnist-original.mat"))
#print(mnist_raw.keys())
#print(mnist_raw)

mnist={
    "data":mnist_raw["data"].T,
    "target":mnist_raw["label"][0]
}
print(mnist['data'].shape)

x,y = mnist["data"],mnist["target"]
number = x[15000]
number_image = number.reshape(28,28)

plt.imshow(
    number_image,
    cmap=plt.cm.binary,
    interpolation="nearest"
)

plt.show()