# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 22:35:54 2018

@author: Jimmy
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

xs = cv2.imread('IMG5.jpg', 1)[:,:,::-1]
# 灰度图
x_gray = cv2.cvtColor(xs, cv2.COLOR_BGR2GRAY)
print(x_gray.shape)
# 亮度翻转
x_i_reverse = 255- x_gray 
# 旋转图像180degree
rows,cols=x_gray.shape
M= cv2.getRotationMatrix2D((cols/2,rows/2),180,1)
print(M.shape)
rotate=cv2.warpAffine(x_gray,M,(cols,rows))
# 加噪声(标准差为0.1的高斯噪声)
imgs = (x_gray-np.min(x_gray))/(np.max(x_gray)-np.min(x_gray))
Grey_gs = imgs + np.random.normal(0, 0.05, imgs.shape)
Grey_gs1 = Grey_gs
# 均衡化
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
equ = clahe.apply(x_gray)
# 畸变图
mtx = np.array([[484.869,0,326.253],[0,496.41,266.175],[0,0,1]])
#dist = np.array([[0.07143217,0.067237,0,0,0.06208967]])
dist = np.random.uniform(0,0.1,(1,5))

# opencv中背景色的设置
h, w = xs.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),0,(w,h))
newcameramtx2, roi2=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),-1,(w,h))
# undistort
#dist = np.random.uniform(0,1,(1,5))
dst = cv2.undistort(xs, mtx, dist, None, newcameramtx)
# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
print(dst.shape)
dst_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

'''
dst2 = cv2.undistort(xs, mtx, dist, None, newcameramtx2)
# crop the image
x2,y2,w2,h2 = roi2
dst2 = dst[y2:y2+h2, x2:x2+w2]
'''

plt.subplot(231),plt.imshow(xs)
plt.xticks([]), plt.yticks([])
plt.subplot(232),plt.imshow(x_gray, cmap ='gray')
plt.xticks([]), plt.yticks([])
plt.subplot(233),plt.imshow(x_i_reverse, cmap ='gray')
plt.xticks([]), plt.yticks([])
plt.subplot(234),plt.imshow(rotate, cmap ='gray')
plt.xticks([]), plt.yticks([])
plt.subplot(235),plt.imshow(Grey_gs, cmap ='gray')
plt.xticks([]), plt.yticks([])
plt.subplot(236),plt.imshow(dst_gray, cmap ='gray') 
plt.xticks([]), plt.yticks([])
'''
plt.subplot(337),plt.imshow(dst, cmap ='gray')
plt.xticks([]), plt.yticks([])
plt.subplot(338),plt.imshow(dst2, cmap ='gray')
plt.xticks([]), plt.yticks([])'''
plt.show()