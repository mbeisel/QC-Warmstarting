from qutip import *
import numpy as np


b = Bloch()
x = (basis(2,0)+(1+0j)*basis(2,1)).unit()
y = (basis(2,0)+(0+1j)*basis(2,1)).unit()
z = (basis(2,0)+(0+0j)*basis(2,1)).unit()
b.add_states([x,y,z])

x = [np.sin(th) for th in np.linspace(0, np.pi/3, 20)]
y = np.zeros(20)
z = [np.cos(th) for th in np.linspace(0, np.pi/3, 20)]
b.add_points([x, y, z])

x = [np.sin(th) for th in np.linspace(np.pi/3, np.pi*2/3, 20)]
y = np.zeros(20)
z = [np.cos(th) for th in np.linspace(np.pi/3, np.pi*2/3, 20)]
b.add_points([x, y, z])

x = [np.sin(th) for th in np.linspace(np.pi*2/3, np.pi*5/3, 60)]
print(x)
# x = np.zeros(20)
# y = [np.sin(th) for th in np.linspace(0, np.pi*1/3, 10)] + [np.sin(th) for th in np.linspace(np.pi*1/3, 0, 10)]
y = np.array([np.sin(th) for th in np.linspace(0, np.pi, 60)]) * 2/3
print(y)
# y = np.zeros(20)
z = [np.cos(th) for th in np.linspace(np.pi*2/3, np.pi*2/3, 60)]
b.add_points([x, y, z])



x = [np.sin(th) for th in np.linspace(np.pi*5/3, 2*np.pi, 20)]
y = np.zeros(20)
z = [np.cos(th) for th in np.linspace(np.pi*2/3, np.pi, 20)]
b.add_points([x, y, z])


b.show()
