"""
last edit: 11/23
object 1 = earth
object 2 = debris
object 3 = Sat.

UNDERSTAND MY DT0 =350S
"""
import matplotlib.pyplot as plt
import numpy as np
import math as M
import time as d#CD
import csv


#def constants and variables
pi = np.pi
km = 1
m = km/1000#CD
kg = 1
sec = 1
newtons = (kg*m)/(sec**2)
G = (6.67E-11)*newtons*(m**2)/(kg**2)
dt = 10*sec###
degree = pi/180
scale = 200*km
shift = 200

m1 = 5.9722E24*kg
m2 = 2501*kg# medium sat size
m3 = 2501*kg
r1 = 6378.137*km
r2 = 2*m
r3 = 2*m
t = 0
a = 0

"""
e-eccentricy
A-semi major
Rp = radius of Perigee(short approch)
c-distace between foci
ra_node- right ascention node
TA = true anomaly
i = inclination(degrees)
w = Argument of periapsis(distance between ra-node and rp)[15]
z =(r*sin(w+TA)*sin(i))
dt chck262 n
    * DT = angle/mean motion
    * time since last perigee = Mean_anomaly/Meanmotions 
"""

fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.set(xlim=(-4*10**4, 4*10**4), ylim=(-4*10**4, 4*10**4), zlim=(-4*10**4, 4*10**4))

row = 0
read_path = "space_decay.csv"
with open(read_path, "r+") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter = ',')
    next(csv_reader)
    for columb in csv_reader:
        write_path = f'{a}.csv'#f'/Users/mario/Dropbox/XimenasWork/SuperComputing2025/derbis/{a}.csv'
        T = float(columb[26])#minutes
        e = float(columb[12])#unitless
        A = float(columb[25])#meters
        i = float(columb[13])#degree
        w = float(columb[15])#degree
        #mean_motion = float(columb[11])#cd
        #mean_anomaly = float(columb[16])#cd
        ra_node = float(columb[14])#degree
        c = e*A#meters
        Rp = A-c#meters
        E_sig = -G*m1*m2/(2*A)#J
        
##        print('t',T)
##        print('e',e)
##        print('a',A)
##        print('i',i)
##        print('ra_node',ra_node)
##        print('w',w)

        theta_z = w*degree
        rotation_matrixs_z = np.array([[np.cos(theta_z), np.sin(theta_z),0],
                                       [-np.sin(theta_z), np.cos(theta_z),0],
                                       [0,0,1]])

        theta_x = i*degree
        rotation_matrixs_x = np.array([[1,0,0],
                                        [0,np.cos(theta_x),np.sin(theta_x)],
                                        [0,-np.sin(theta_x),np.cos(theta_x)]])

        theta_z2 = ra_node*degree
        rotation_matrixs_z2 = np.array([[np.cos(theta_z2), np.sin(theta_z2),0],
                                        [-np.sin(theta_z2), np.cos(theta_z2),0],
                                        [0,0,1]])

##        ax.scatter(0,0,0, ls='-', label='earth')        
        R = [Rp, 0, 0]
        r_oldish = np.matmul(rotation_matrixs_z, R)
        r_oldishish = np.matmul(rotation_matrixs_x, r_oldish)
        r_old = np.matmul(rotation_matrixs_z2, r_oldishish)

        time = 0
        for TA in range(1,360):
            
            radius = A*(1-(e**2))/(1+e*np.cos(TA*degree))#tA as a function ta|0-360

            #r_old = np.array([50000*np.cos(TA*degree),50000*np.sin(TA*degree),0])
            R = np.array([radius*np.cos(TA*degree),radius*np.sin(TA*degree),0])

            #rotate by the R.M.s
            r_newish = np.matmul(rotation_matrixs_z, R)
            r_newishish = np.matmul(rotation_matrixs_x, r_newish)
            r_new = np.matmul(rotation_matrixs_z2, r_newishish)

            
            #find DT
            Ue = -G*m1*m2/radius
            velocity = M.sqrt(2*(E_sig-Ue)/m2)

            distance = np.linalg.norm(r_new-r_old)
            dt = distance/velocity

            #save data
            f = open(write_path,'a')
            L = f'{time},{r_new[0]},{r_new[1]},{r_new[2]},\n'
            f.writelines(L)
            f.close()
            
##            #plot points 
            if TA%5 == 0:
                ax.scatter(r_new[0], r_new[1],r_new[2],ls='-', label = 'debris')
                ax.scatter(r_old[0], r_old[1],r_old[2],c='green', label = 'debris')
            time += dt
            r_old = r_new 
        plt.show()
        a += 1
        

print('done')
