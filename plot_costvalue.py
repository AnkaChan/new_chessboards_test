# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 14:58:35 2018

@author: Jimmy
"""

import numpy as np
import matplotlib.pyplot as plt
m = 2000
a = np.loadtxt("cost_value4.txt")[:m]
#b = np.loadtxt("cost_value5.txt")[:m]
#c = np.loadtxt("cost_value6.txt")[:m]
plt.plot(a)
#plt.plot(b)
#plt.plot(c)
plt.show()

