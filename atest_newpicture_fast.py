# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 22:19:54 2018

@author: Jimmy
"""

# 5是最老的一般，在原始的数据集上训练10万次的结果
# 6是将costfuction的系数从56降到40训练的一款
# 7是系数为40，学习率为i0.05, 正则化系数为0.0005的训练2万次的结果
# 本程序用于测试非自己采集的数据
import tensorflow as tf
import numpy as np
import cv2
import time
from nms_fast import non_max_suppression_fast
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
# 加载 chessboards_infernece.py 中定义的常量和前向传播函数
import new_chessboards_infernece

# 定义测试函数
def evaluate(xs,reshaped_xs):
    g = tf.Graph()
    with g.as_default():
# 定义输出为4维矩阵的placeholder
        x = tf.placeholder(tf.float32, [None, reshaped_xs.shape[1],
                                        reshaped_xs.shape[2], 1], name='x-input')

# 直接调用封装好的函数来计算前向传播结果。因为测试时不关心正则化损失的值，所以
# 这里用于计算正则化损失的函数被设置为 None
        y = new_chessboards_infernece.inference(x, None, None)

# 通过变量重命名的方式来加载模型，这样就可以在前向传播的过程中就不需要调用求滑动平均
# 的函数来获取平均值了。这样就可以完全共用 mnist_infernece.py 中定义的前向传播过程
        variable_averages = tf.train.ExponentialMovingAverage(0.99)
        variables_to_restore = variable_averages.variables_to_restore()

# 初始化TensorFlow的持久化类
# variables_to_restore已将所有的trainable的变量使用滑动平均值代替
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            
# 加载已经保存的模型
            saver.restore(sess, "p3chessboards_model/p3chessboards_model")

# 调用评估数据集中的值（如果计算测试准确率，可以调用测试数据值）
            y = sess.run(y, feed_dict={x:reshaped_xs})
            start = time.clock()
            reshaped_y = np.reshape(y, (reshaped_xs.shape[1], reshaped_xs.shape[2]))
#            print(reshaped_y.max())
#            yy = np.select([reshaped_y<1, reshaped_y>=1], [reshaped_y,1])
#            equ = yy.copy()            
#            imgs = np.int0(np.round((equ-np.min(equ))/(np.max(equ)-np.min(equ))*255))
#            histr = cv2.calcHist([np.uint8(imgs)],[0],None,[256],[0,256])
#            #print(histr)
#            plt.plot(histr[1:])
#            plt.show()
# 输出图像NMS前角点
            ret,mm = cv2.threshold(reshaped_y, 0.6*np.amax(reshaped_y),1, cv2.THRESH_BINARY)
            print("[x] before applying non-maximum, %d bounding boxes" %(int(np.sum(mm))))
# 设置非极大值抑制窗口半径
            #m = np.zeros((reshaped_xs.shape[1], reshaped_xs.shape[2]))
            boxes = np.transpose(np.nonzero(mm))      
# 对图像进行非极大值抑制
            pick = non_max_suppression_fast(boxes, 0.5)
            print ("[x] after applying non-maximum, %d bounding boxes" % (len(pick)))
            y_pred = KMeans(n_clusters=10, random_state=0).fit_predict(pick).tolist()
            #print(y_pred)
            print(pick.shape)
            """
### 方法1
            zz = []
            for i in range(10):
                if y_pred.count(i)>1:
                    zz.append(i)
            #cv2.circle(xs,(pick[i][1],pick[i][0]), 3, [0,0,255], -1)
            
            [cv2.circle(xs,(pick[i][1],pick[i][0]), 3, [0,0,255], -1) for i in range(len(pick)) if y_pred[i] in zz]
            """
### 方法2
            pick = np.delete(pick,[i for i in range(len(pick)) if y_pred.count(y_pred[i])<=2],axis=0)
            
            for i in range(len(pick)):
                cv2.circle(xs,(pick[i][1],pick[i][0]), 3, [0,0,255], -1)
            #pick = np.delete(pick,pick2,axis=0)
            cv2.imwrite("aam1.jpg",xs)
            #end = time.clock()
            print(len(pick))
            end = time.clock()
            print("The time cost is: ", str(end-start))
#主程序
def main(argv=None):
    xs = cv2.imread('t1.jpg', 1)
    ys = cv2.cvtColor(xs, cv2.COLOR_BGR2GRAY)
    #equ = cv2.equalizeHist(ys)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    equ = clahe.apply(ys)
    #imgs = equ/255
    imgs = (equ-np.min(equ))/(np.max(equ)-np.min(equ))
    reshaped_xs = np.reshape(imgs, (1, ys.shape[0], ys.shape[1], 1))
    evaluate(xs,reshaped_xs)

if __name__ == '__main__':
    main()