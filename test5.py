# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 10:54:47 2019

@author: Thinker
"""

import os
name_list=os.listdir('E:/python/cnn/faceimages/')
labels_dic={}
for i in range(len(name_list)):
    labels_dic[name_list[i]]=i
    labels_dic[i]=name_list[i]

#加载人脸识别模型
import tensorflow as tf
import numpy as np
tf.reset_default_graph()#重置计算图

saver = tf.train.import_meta_graph('E:/python/cnn/model_10/train_model.meta')#读取模型中的meta文件--导入计算图(meta)
graph = tf.get_default_graph()
x_new = graph.get_tensor_by_name('x_date:0')
y_new = graph.get_tensor_by_name('pre:0')
keep_prob= graph.get_tensor_by_name('keep_prob:0')

#人脸识别    
def face_(data=np.zeros([1,32,32,3],dtype='float32')):
    with tf.Session() as sess:
        saver.restore(sess,tf.train.latest_checkpoint('E:/python/cnn/model_10/'))#导入计算图的所以参数
        
        pre = sess.run(y_new,feed_dict={x_new:data,keep_prob: 1.0})#计算预测值

    pre_=np.max(pre)#获取预测概率
    pre=np.argmax(pre)#获取预测值
    name=labels_dic[pre]#将预测值转译为名字
    return name,pre_

import cv2
import tensorflow as tf
import detect_face 

#初始化mtcnn人脸检测算法
minsize = 20 # minimum size of face 
threshold = [ 0.6, 0.7, 0.7 ] # three steps's threshold 
factor = 0.709 # scale factor 
gpu_memory_fraction=1.0

with tf.Graph().as_default(): 
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction) 
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)) 
    with sess.as_default(): 
      pnet, rnet, onet = detect_face.create_mtcnn(sess, None) 
      
font = cv2.FONT_HERSHEY_SIMPLEX
x1=x2=y1=y2=0
#人脸检测
def Detect_face(img):
    face_id, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor) 
    
    if face_id.shape[0]>0 and face_id[0].shape[0]>0:
        face_box=face_id[0]
        if face_box[0]>0 and face_box[1]>0:
            face_box=face_box.astype(int)
            x1,y1,x2,y2=face_box[0],face_box[1],face_box[2],face_box[3]
            cv2.rectangle(img,(x1, y1), (x2, y2), (0, 255, 0), 2)#绘制人脸矩形边框
                
            n_img=img[y1:y2,x1:x2,0]
            n_img=cv2.resize(n_img,(32,32))#重设图片大小

            n_img=np.float32(np.reshape(n_img,[1,32,32,1]))
            name,pre=face_(n_img)
            pre='%.4f'%(pre*100)
    else:
        name='Unknow'
        pre='0.0000%'
    return name,pre

      
my_video=cv2.VideoCapture(0)
while(True):
    retval, img=my_video.read()
    name, pre = Detect_face(img) 
    cv2.putText(img, name+' '+pre+'%', (50, 50), font, 1.2, (255, 255, 0), 2)
    cv2.imshow('image',img)
    if(cv2.waitKey(1)==27):
        cv2.destroyAllWindows()
        break
my_video.release()
