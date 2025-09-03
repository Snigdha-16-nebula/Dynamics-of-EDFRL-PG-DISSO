#Lasing intensity vs time




import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
s,g,alpha0,d,a,v=0.26,1.1*10e5,12.2,0.285,23.37,1.1*10e3
#tau_s=0.01
m=0.0012 #modulation index
#f= float(input('frequency='))#modulation frequency
#f=float(14.4)
nu=117.8#scaled frequency=(f in kHz*tau_s)
theta=np.linspace(0,10,1000000)#time array
#theta=t/tau_s #time array
alpha_m=m*np.sin(2*np.pi*nu*theta) #time varying part of modulation frequency with modulation index
alpha=alpha0+m*np.sin(2*np.pi*nu*theta) #modulation frequency
#solving with odeint function
def shm(z,theta):
    x,y=z[0],z[1]
    alpha=alpha0+float(m*np.sin(2*np.pi*nu*theta))
    dxdtheta=s+g*x*(y-alpha) #change in scaled photon density with scaled time
    dydtheta=a-y*(x+1+d)     #change in population inversion wrt scaled time
    return [dxdtheta,dydtheta]
z0=[0.001,0.001] #initial values

sol=odeint(shm,z0,theta)
#extract the solutions for x and y
x,y=sol[:,0],sol[:,1]
#PLOT
plt.plot(theta,x,'b')
plt.plot(theta,alpha_m,'k')
plt.ylabel ('Scaled Lasing intensity(x)',size='20')
plt.xlabel ('Scaled time(theta)',size='20')
plt.title(' Scaled Lasing intensity(x) vs Scaled time(theta)')
#plt.xlim(0.8,1.5)
plt.grid(True)
plt.show()
