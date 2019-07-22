# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#from mtcnn.mtcnn import MTCNN
import cv2
import os

#初始化人脸检测函数
import detect_face 
import tensorflow as tf

minsize = 20 # minimum size of face 
threshold = [ 0.6, 0.7, 0.7 ] # three steps's threshold 
factor = 0.709 # scale factor 
gpu_memory_fraction=1.0

with tf.Graph().as_default(): 
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction) 
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)) 
    with sess.as_default(): 
      pnet, rnet, onet = detect_face.create_mtcnn(sess, None) 

faces_list=os.listdir('E:/python/cnn/faceimages')#获取数据集目录

#创建总存储目录
if os.path.exists('E:/python/cnn/faceimagesGrey')==False:
    os.mkdir('E:/python/cnn/faceimagesGrey')
    
for i in range(len(faces_list)):
    r_path='E:/python/cnn/faceimages/'+faces_list[i]+'/' #读取路径
    w_path='E:/python/cnn/faceimagesGrey/'+faces_list[i]+'/' #写入路径
    images_list=os.listdir(r_path) #图像集目录
    
    #创建存储文件夹
    if os.path.exists(w_path)==False:
        os.mkdir(w_path)
    
    for j in range(600):
    
        img = cv2.imread(r_path+images_list[j])#读取图像
        face_id=detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)#通过mtcnn算法获取脸部位置信息
        
        #通过mtcnn算法可检测出人脸（无法检测的时候会返回[]）
            #在无法检测的情况下，沿用上次的可识别数据
        if face_id.shape[0]!=0 and face_id[0].shape[0]!=0:
            face_box=face_id[0][0]
            if face_box[0]>0 and face_box[1]>0:
                face_box=face_box.astype(int)
                x1,y1,x2,y2=face_box[0],face_box[1],face_box[2],face_box[3]
                if x2-x1>100:
                    n_img=img[y1:y2,x1:x2,0] #截取人脸部分并将图片灰化处理       
        cv2.imwrite(w_path+images_list[j],n_img) #存储处理完毕后的图像到指定路径
        
        #打印运行进度
        if j%100==0 :
            print(i,j) 