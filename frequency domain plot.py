import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Define the system of differential equations
def system(y, t, f):
    x, y = y  # Unpack the state vector
    neu=f*10
    ap=12.2*float(1+0.0012*np.sin(2*np.pi*float(neu)*t))

    dxdt = 0.26+1.11e5*x*(y-ap) # Example equation using frequency
    dydt = 23.37-y*(x+1+0.285) # Example equation using frequency
    return [dxdt, dydt]

# Initial conditions
initial_conditions = [0.001,0.001]

# Time points
t = np.arange(0, 10, 1e-6)

# Frequency (value in kHz)
frequency =13.9


# Solve the system of equations
solution = odeint(system, initial_conditions, t, args=(frequency,))

# Extract the solution for x
x_solution = solution[:, 0]

# Perform the FFT on the x solution
fft_x = np.fft.fft(x_solution)
power_spectrum = np.abs(fft_x)/1e6

# Get the corresponding frequencies in Hz
frequencies = np.fft.fftfreq(len(t), d=1e-6)

# Plot the power versus frequency graph
plt.figure(figsize=(10, 5))
plt.plot(frequencies/10, power_spectrum)  # Convert frequency to kHz
#plt.xlim(0, 500)  # Set x-axis limit to 500 kHz for better visibility
plt.xlabel('Frequency (kHz)',size='20')
plt.ylabel('Magnitude',size='20')
plt.title('Frequency Domain',size='20')
plt.grid(True)
plt.xlim(0,40)
plt.show()
