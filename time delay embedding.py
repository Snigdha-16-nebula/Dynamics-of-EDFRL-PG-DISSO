import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint 
from mpl_toolkits.mplot3d import Axes3D

def time_delay_embedding(data, embedding_dimension, tau):
    """
    Perform time delay embedding on input data.
    
    Parameters:
        data (array-like): 1D array or time series data.
        embedding_dimension (int): Dimension of the embedded space.
        time_delay (int): Time delay between consecutive samples.
        
    Returns:
        np.array: Time-delay embedded data matrix.
    """
    n = len(data)
    embedded_data = np.zeros((n - (embedding_dimension - 1) * tau, embedding_dimension))
    
    for i in range(embedding_dimension):
        embedded_data[:, i] = data[i * tau : i * tau + len(embedded_data)]
        
    return embedded_data

# Define the coupled differential equations
s,g,alpha,d,a=0.26,1.11*10e5,12.19,0.285,23.37
def shm(z,theta,f):
    x,y=z[0],z[1]
    #f=12.3
    neu=f*10

    #sinusoidally varying variable
    ap=12.2*float(1+0.0012*np.sin(2*np.pi*neu*theta))

    #diff eqns
    dxdt = 0.26 + (1.11e5) * x * (y - ap)
    dydt = 23.37 - y * (x + 1 + 0.285)
    return [dxdt,dydt]

z0=[0.001,0.001] # Initial conditions


# Time points
theta = np.arange(0, 10, 0.000001)

# Frequency of sinusoidally varying variable a
f = 13.9

# Solve the differential equations
sol = odeint(shm, z0, theta, args=(f,))

x,y=sol[:,0],sol[:,1]
print("x:",x)
data=x  # Example time series data (the solution array x obtained by solving differential eqns.)
embedding_dimension = 3  # Embedding dimension
tau = 100# Time delay between consecutive samples

embedded_data = time_delay_embedding(data, embedding_dimension, tau)
print("Embedded Data Shape:", embedded_data.shape)
#print("Embedded Data:")
#print(embedded_data)
def plot_embedded_data(embedded_data):
    """
    Plot time-delay embedded data in 3D.
    
    Parameters:
        embedded_data (array-like): Time-delay embedded data matrix.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(embedded_data[:, 0], embedded_data[:, 1], embedded_data[:, 2], c='b', marker='o', markersize=1)
    ax.set_xlabel('X(t)',size='20')
    ax.set_ylabel('X(t - tau)',size='20')
    ax.set_zlabel('X(t - 2*tau)',size='20')
    ax.set_title('Time Delay Embedding',size='20')

    plt.show()

# Example usage
plot_embedded_data(embedded_data)
