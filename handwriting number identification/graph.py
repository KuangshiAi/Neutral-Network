import numpy
import scipy.special
import matplotlib.pyplot
import math
f=open(r'C:\Users\艾志敏\Desktop\程序设计\Python\mnist_10_test.csv','r')
#注意不能使用readline，否则是一个个字符来读取的
list=f.readlines()
f.close()
all_value=list[9].split(',')
image=numpy.asfarray(all_value[1:]).reshape(28,28)
matplotlib.pyplot.imshow(image,cmap='Greys',interpolation='none')
matplotlib.pyplot.show()
