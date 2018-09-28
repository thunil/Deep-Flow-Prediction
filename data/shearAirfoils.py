################
#
# Deep Flow Prediction - N. Thuerey, K. Weissenov, H. Mehrotra, N. Mainali, L. Prantl, X. Hu (TUM)
#
# Shear airfoils to produce more variance
#
################

import numpy as np
import os
import uuid
from utils import saveAsImage
import matplotlib.pyplot as plt
from random import randint

airfoil_database = "./airfoil_database/"
output_dir = "./airfoil_database_sheared/"
shear = np.identity(2)
shear[0,1] = np.random.uniform(0.95, 1.05)

files = os.listdir(airfoil_database)
samples = len(files)
for n in range(samples):
    print("Run {}:".format(n))

    shear[0,1] = np.random.uniform(0.95, 1.05)
    print("\tusing {} , shear {}".format(files[n], shear[0,1]))

    airfoilFile = airfoil_database + files[n]
    arf = np.loadtxt(airfoilFile, skiprows=1)
    ar = arf.copy()

    tempar = ar.copy()
    arshear = np.dot(shear, ar.T).T

    maxx_original= max(arshear[:,0])
    maxx_shear = max(arf[:,0])
    maxy_original= max(arshear[0,:])
    maxy_shear= max(arf[0,:])
    for i in range(len(arshear)):
        arshear[i,0] = arshear[i,0]*(maxx_original/maxx_shear)
        arshear[i,1] = arshear[i,1]*(maxy_original/maxy_shear)

    if 0:
        plt.subplot(1, 2, 1)
        a, b = ar.T
        plt.scatter(a,b)
        plt.axis('equal')
        plt.subplot(1, 2, 2)
        c, d = arshear.T
        plt.scatter(c,d)
        plt.axis('equal')
        #plt.show()

    basename = os.path.splitext( os.path.basename(files[n]) )[0]
    nid = "_%d" % int(shear[0,1]*1000.)
    fn = output_dir+basename+nid+".dat"
    print("\twriting {} ".format(fn))
    np.savetxt(fn, arshear, header = files[n] + ' shear')

