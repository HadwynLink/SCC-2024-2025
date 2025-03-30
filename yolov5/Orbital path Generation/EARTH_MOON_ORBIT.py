import math as M
import csv
import matplotlib.pyplot as plt
import numpy as np

pi = np.pi
kg = 1
km = 1
m = km/1000
sec = 1
newtons =(kg*m)/(sec**2)
G = (6.67E-11)*newtons*(m**2)/(kg**2)
degree = pi/180
dt = 100*sec

m1 = 5.9722E24 *kg#earth
m2 =  7.34767309E22 *kg#moon
r1 = 0
r2 = 384_400 *km
T1 = 0#in theory
T2 = M.sqrt(4*(pi**2)*(r2**3)/(m1*G))

path = '/Users/mario/Dropbox/XimenasWork/SuperComputing2025/earth_moon.csv'

##change limits
##fig = plt.figure()
##ax = plt.axes(projection = '3d')
##ax.set(xlim=(-4*10**5, 4*10**5), ylim=(-4*10**5, 4*10**5), zlim=(-4*10**5, 4*10**5))
##ax.set_xlabel('X')
##ax.set_ylabel('Y')
##ax.set_zlabel('Z')

r1_old = np.array([0,0,0])

r2_old = ([r2,0,0])
t = 0

#ax.scatter(0, 0, 0, c='green')
while t < T2:
    
    r2_new  = np.array([r2*np.cos(t*2*pi/T2),r2*np.sin(t*2*pi/T2),0])
    #ax.scatter(r2_new[0], r2_new[1], r2_new[2], c='blue')

    f = open(path,'a')
    L = [f'{t}, {r2_new[0]}, {r2_new[1]}, {r2_new[2]},\n']#,{r2_new}, \n']
    f.writelines(L)
    f.close()
    t += dt
#plt.show()    

print('done')
