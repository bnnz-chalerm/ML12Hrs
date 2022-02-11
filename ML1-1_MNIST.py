import matplotlib.pyplot as plt
from sklearn import datasets
## load data
digit_dataset = datasets.load_digits()

# print(digit_dataset.keys())
# #print(digit_dataset['DESCR'])
# print(digit_dataset['images'].shape)
# print(digit_dataset.target_names)
# print(digit_dataset.images[0])
# print(digit_dataset.images[:15])
print(digit_dataset.target[:10])
plt.imshow(digit_dataset.images[10],cmap=plt.get_cmap('gray'))
plt.show()