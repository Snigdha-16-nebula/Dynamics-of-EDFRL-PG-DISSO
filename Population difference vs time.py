import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
s=0.26 #scaled spontenous emission rate
g=1.1*10e5 #ratio of tau_s and tau_r
tau_s=0.01 #tau_s=upper state life time
alpha_0=12.2 #cavity loss
d=0.285 #pumping rate dependent constant
a=23.37 #pumping rate dependent constant
m=0.0012 #modulation index
# Frequency of sinusoidally varying variable a
f = 13.9
nu=f*10
def shm(z,theta,f):
    x,y=z[0],z[1]
    #sinusoidally varying variable
    alpha=12.2*float(1+0.00001*np.sin(2*np.pi*nu*theta))

    #diff eqns
    dxdt = 0.26 + (1.11e5) * x * (y - alpha)
    dydt = 23.37 - y * (x + 1 + 0.285)
    return [dxdt,dydt]

z0=[0.001,0.001] # Initial conditions


# Time points
t = np.arange(0, 10, 0.000001)



# Solve the differential equations
sol = odeint(shm, z0, t, args=(f,))
x=sol[:,0]
y=sol[:,1]
#print(np.shape(alpha))
print(np.shape(t))
# Plot the solutions
plt.figure(figsize=(10, 6))
#plt.plot(t[1000:10000000], x[1000:10000000], 'b', label='x(t)')
plt.plot(t,y,label='x(t)')
#plt.plot(t,ap)
#plt.plot(t, sol[:, 1], 'g', label='y(t)')
plt.title('Phase portrait obtained by solving coupled differential Equations for modulation frequency=10kHz')
plt.xlabel('scaled time(theta)',size='20')
plt.ylabel('Scaled population difference')
plt.legend()
plt.text(0,12,"f=10kHz")
plt.ylim(12.17,12.24)
plt.grid()
plt.show()
