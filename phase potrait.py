
#tau_r=round trip time,g=tau_s/tau_r
#tau_s=upper level lifetime
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
#s,g,alpha0,d,a,v,alpha_m=0.26,1.1*10e5,12.2,0.285,23.37,1.1*10e3,3.2*10e-3
s=0.26 #scaled spontenous emission rate
g=1.11*10e5 #ratio of tau_s and tau_r
tau_s=0.01 #tau_s=upper state life time
alpha_0=12.2 #cavity loss
d=0.285 #pumping rate dependent constant
a=23.37 #pumping rate dependent constant
m=0.0012 #modulation index
f= float(input('frequency='))#modulation frequency
#f=float(14.4)
nu=(f)*10#scaled frequency=(f in kHz*tau_s)

#theta=np.linspace(0.5,1.5,9000)
t=np.linspace(0.1,10,500000)#time array
theta=t/tau_s #time array
alpha_m=m*np.sin(2*np.pi*nu*theta) #time varying part of modulation frequency with modulation index
alpha=alpha_0+m*np.sin(2*np.pi*nu*theta) #modulation frequency
#solving with odeint function
def shm(z,theta):
    x,y=z[0],z[1]
    alpha=alpha_0+float(m*np.sin(2*np.pi*nu*theta))
    dxdtheta=s+g*x*(y-alpha) #change in scaled photon density with scaled time
    dydtheta=a-y*(x+1+d)     #change in population inversion wrt scaled time
    return [dxdtheta,dydtheta]
z0=[0.001,0.001] #initial values
sol=odeint(shm,z0,theta)
#extract the solutions for x and y
x,y=sol[:,0],sol[:,1]
#PLOT
plt.plot(theta,x,'b')#
plt.plot(theta,alpha_m,'k')
plt.xlabel ('Scaled time(theta)',size='20')#'Scaled Lasing intensity(x)',size='20')
plt.ylabel('Scaled Lasing intensity(x)',size='20')#('Scaled Lasing intensity(x)')
plt.title('Scaled time(theta) vs  Scaled Lasing intensity(x)',size='20')#(' Scaled Lasing intensity(x) vs Population difference',size='20') #Scaled time(theta)Scaled Lasing intensity(x)')
#plt.text(8,7,('m=0.00001 and f=14 kHz'))
#plt.plot(theta,alpha_m,'b')
#plt.plot(theta,y,'k-')
#y)
#plt.xlim(0,4)
#plt.ylim(5,15)
#plt.pause(1)
plt.grid()
plt.show()
