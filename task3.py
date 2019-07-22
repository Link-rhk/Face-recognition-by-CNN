# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 17:08:54 2019

@author: Thinker
"""
import os
import cv2

Data_path='E:/python/cnn/faceimagesGrey/'#设置数据读取路径
face_list=os.listdir(Data_path)#读取数据集目录

Faceimage={'data':[],'labels':[]}#数据--标签

for i in range(len(face_list)):
    
    label=face_list[i]#设置图片读取路径
    image_list=os.listdir(Data_path+face_list[i])#读取图像集集目录
    
    for j in range(600):
        img=cv2.imread(Data_path+face_list[i]+'/'+image_list[j])#读取图片
        n_img=cv2.resize(img,(32,32))#重设图片大小
        n_img=n_img[:,:,0]#转换为单通道图片
        
        #收集数据与标签
        Faceimage['data'].append(n_img)
        Faceimage['labels'].append(face_list[i])
        
        #汇报执行进度
        if j%100==0:
            print(i,j)

import numpy as np

D_path='E:/python/cnn/faceimagesData/'#设置存储路径

#确认存储文件夹
if os.path.exists(D_path)==False:
    os.mkdir(D_path)

#存储Faceimage    
np.save(D_path+'Faceimage_data.npy',Faceimage['data'])
np.save(D_path+'Faceimage_labels.npy',Faceimage['labels'])
        
from sklearn.model_selection import train_test_split
data_train,data_test,labels_train,labels_test=train_test_split(Faceimage['data'],
                                                               Faceimage['labels'],
                                                               test_size=0.2) #随机分划训练集和测试集
#存储训练集、测试集
np.save(D_path+'data_train.npy',data_train)
np.save(D_path+'data_test.npy',data_test)
np.save(D_path+'labels_train.npy',labels_train)
np.save(D_path+'labels_test.npy',labels_test)

             
