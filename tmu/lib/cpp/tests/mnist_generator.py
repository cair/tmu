from tmu.data import MNIST
import numpy as np
data = MNIST().get()
print(data.keys())

x_train = data["x_train"]
y_train = data["y_train"]
x_test = data["x_test"]
y_test = data["y_test"]

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# 2d, save as files whhere value is separated by space
np.savetxt("x_train.txt", x_train, fmt="%d")
np.savetxt("y_train.txt", y_train, fmt="%d")
np.savetxt("x_test.txt", x_test, fmt="%d")
np.savetxt("y_test.txt", y_test, fmt="%d")
