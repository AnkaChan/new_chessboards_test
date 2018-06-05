# -*- coding: utf-8 -*-
"""
Created on Thu May 11 11:11:02 2017

@author: benchen4395
"""

import numpy as np
import matplotlib.pyplot as plt
# 读取均方差代价函数
fopen2 = open('shujuMSE.txt','r')
LineArr2 = []
for line2 in fopen2.readlines():
    currline2 = str(line2.strip().split('\t')).strip('[').strip(']').strip("'").split(' ')
    currline2 = list(currline2)

    LineArr2.append(currline2[-1][:-1])
fopen2.close()
# 读取交叉熵代价函数
fopen = open('shuju.txt','r')
LineArr = []
for line in fopen.readlines():
    currline = str(line.strip().split('\t')).strip('[').strip(']').strip("'").split(' ')
    currline = list(currline)

    LineArr.append(currline[-1][:-1])
fopen.close()
c = list(range(1,2001))
a = LineArr[:-1].copy()
b = LineArr2[:-1].copy()
for i in range(2000):
    a[i] = float(a[i])
    b[i] = float(b[i])
#d = np.ones_like(a)*0.05
plt.plot(c[:300],a[:300],'r',label="Cross Entropy")
plt.plot(c[:300],b[:300],'b',label="MSE")
#plt.plot(c[:300],d[:300],'k',label="constant")
plt.legend(loc='upper right',fontsize='x-large')
#plt.title("Cost function", fontsize=24)
plt.xlabel("Training time (epochs)", fontsize=16)
plt.ylabel("MSV", fontsize=16)
plt.tick_params(axis='both', labelsize=14)
#plt.show()
plt.savefig('cost function1.jpg', bbox_inches='tight')
