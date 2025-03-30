import math as M
import time
import pygame as pg#cd
import numpy as np
import matplotlib.pyplot as plt
import pygame as pg
import csv

pi = np.pi
kg = 1
km = 1
m = km/1000
sec = 1
hour = 60*60*sec
newtons =(kg*m)/(sec**2)
G = (6.67E-11)*newtons*(m**2)/(kg**2)
dt = 100*sec
degree = pi/180
shift = 100
scale = 9000*km

m1 = 5.9736E24*kg
m2 = 6200*kg
r1 = 0
r2 = 2_000*km + 6_370*km
T1 = 0#in theory
T2 = M.sqrt(4*(pi**2)*(r2**3)/(m1*G))

fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.set(xlim=(-4*10**5, 4*10**5), ylim=(-4*10**5, 4*10**5), zlim=(-4*10**5, 4*10**5))
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.ion()
plt.show()

path = '/Users/mario/Dropbox/XimenasWork/SuperComputing2025/circular_plot.csv'

r1_old = np.array([0,0,0])
v1_old = np.array([0,0,0])

r2_old = ([r2,0,0])
t = 0
ax.clear()
ax.scatter(0, 0, label = 'debris',)
while t < T2:
    r2_new = np.array([r2*np.cos(t*2*pi/T2),r2*np.sin(t*2*pi/T2)])
    ax.legend()
    plt.draw()
    plt.pause(0.0001)
    f = open(path,'a')
    L = [f'{t}, ',f'{r2_new}','\n']
    #f.writelines(L)
    f.close()
    r2_old = r2_new
    t+= dt
