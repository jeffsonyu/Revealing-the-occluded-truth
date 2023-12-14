import numpy as np
import matplotlib.pyplot as plt
from serial_broker.Conversion import VoltageToForceParameters, voltage2force, force2voltage

test_x = np.arange(0,4,0.2)
default_params = VoltageToForceParameters(A=3.91487, b=1.4283, k=0.75369, d=0.20016)
print("x=", test_x)
f_x = voltage2force(test_x, default_params)
print("f(x)=", f_x)
plt.plot(test_x, f_x)
plt.show()
f_1_x = force2voltage(f_x, default_params)
print("f^-1(x)=", f_1_x)
plt.plot(f_x, f_1_x)
plt.show()
