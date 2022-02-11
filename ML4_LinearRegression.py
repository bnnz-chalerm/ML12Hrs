#y = ax + b
import numpy as np
import matplotlib.pyplot as plt

x= np.linspace(-5,5,100)
#print(x)

y = 2*x+1

#แสดงแบบ plot
#plt.plot(x,y,'-r',label='y=2x+1')
#แสดงแบบการกระจาย
plt.scatter(x,y)
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc="upper left")
plt.title("Graph y=2x+1")
plt.grid()
plt.show()