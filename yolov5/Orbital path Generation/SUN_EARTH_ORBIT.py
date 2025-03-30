import math as M
import csv
import matplotlib.pyplot as plt
import numpy as np

pi = np.pi
kg = 1
km = 1
m = km/1000
sec = 10
newtons =(kg*m)/(sec**2)
G = (6.67E-11)*newtons*(m**2)/(kg**2)
degree = pi/180
dt = 100*sec

m1 = 1988400E24 *kg #sun
m2 =  5.9722E24 *kg
r1 = 0
r2 = 1.496E8
T1 = 0#in theory
T2 = M.sqrt(4*(pi**2)*(r2**3)/(m1*G))

path = '/Users/mario/Dropbox/XimenasWork/SuperComputing2025/sun_earth.csv'

##change limits
##fig = plt.figure()
##ax = plt.axes(projection = '3d')
##ax.set(xlim=(-2*10**8, 2*10**8), ylim=(-2*10**8, 2*10**8), zlim=(-2*10**8, 2*10**8))
##ax.set_xlabel('X')
##ax.set_ylabel('Y')
##ax.set_zlabel('Z')

r1_old = np.array([0,0,0])
v1_old = np.array([0,0,0])

r2_old = ([r2,0,0])
t = 0

#ax.scatter(0, 0, 0, c='green')
while t < T2:
    r2_new  = np.array([r2*np.cos(t*2*pi/T2),r2*np.sin(t*2*pi/T2),0])
    shifted_sun = np.array([0-r2_new[0], 0-r2_new[1], 0-r2_new[2]])

    #ax.scatter(0-r2_new[0], 0-r2_new[1], 0-r2_new[2], c='yellow')

    f = open(path,'a')
    L = [f'{t},{shifted_sun[0]},{shifted_sun[1]},{shifted_sun[2]},\n']
    f.writelines(L)
    f.close()
    
    r2_old = r2_new
    t += dt
    
plt.show()
