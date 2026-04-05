import numpy as np
import matplotlib.pyplot as plt

def manufactured_profiles(r, t, eps):
    Er = 0.4 * np.tanh(8*(r-0.5)) * (1 + 0.1*np.sin(t))
    n = 1 + 0.1*np.cos(np.pi*r) * np.exp(-t) + 0.2*Er
    T = 1.2 - 0.2*r + 0.05*np.sin(2*np.pi*r)*np.exp(-0.5*t) + 0.1*Er
    return n, T, Er

r = np.linspace(0, 1, 200)
t = 0.0
epsilon = 1.0
n, T, Er = manufactured_profiles(r, t, epsilon)

plt.figure(figsize=(8, 5))
plt.plot(r, n, label='n (density)')
plt.plot(r, T, label='T (temperature)')
plt.plot(r, Er, label='Er (radial field)')
plt.xlabel('r')
plt.ylabel('Profile value')
plt.title('Manufactured Profiles at t=0, epsilon=1.0')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
