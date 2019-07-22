# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 21:04:54 2019

@author: Thinker
"""

import os

D_Path='E:/python/cnn/faceimages/'#设置存储目录
name=input('你的名字(拼音):')
#创建相应的文件夹
if os.path.exists(D_Path+name)!=True:
    os.mkdir(D_Path+name)
    
import cv2

my_video=cv2.VideoCapture(0)#打开摄像头
#逐帧进行摄像600次，并将数据存储到 A_image
A_image=[]
for i in range(600):
    retval, image=my_video.read()
    A_image.append(image)

my_video.release()#关闭摄像头

#存储相片到指定目录
for i in range(600):
    j='%d'%i
    cv2.imwrite(D_Path+name+'/'+j+".jpg",A_image[i])