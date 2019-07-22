# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 11:30:53 2019

@author: Thinker
"""

#导入训练数据
import numpy as np
D_Path='E:/python/cnn/faceimagesData/'
data_tr=np.load(D_Path+'data_train.npy')
labels_tr=np.load(D_Path+'labels_train.npy')

data_te=np.load(D_Path+'data_test.npy')
labels_te=np.load(D_Path+'labels_test.npy')

#数据预处理
data_tr=np.float32(np.reshape(data_tr,[data_tr.shape[0],data_tr.shape[1],data_tr.shape[2],1]))
data_te=np.float32(np.reshape(data_te,[data_te.shape[0],data_te.shape[1],data_te.shape[2],1]))

#创建一个名字与数字相互对应的字典
import os
name_list=os.listdir('E:/python/cnn/faceimages/')
labels_dic={}#创建一个名字与数字相互对应的字典
for i in range(len(name_list)):
    labels_dic[name_list[i]]=i
    labels_dic[i]=name_list[i]
Sort_=len(name_list)

#将标签转译为独热编码形式
One_hot_tr=np.zeros([labels_tr.shape[0],Sort_],dtype='float32')
for j in range(labels_tr.shape[0]):
    transform=labels_dic[labels_tr[j]]
    One_hot_tr[j,transform]=1
    
One_hot_te=np.zeros([labels_te.shape[0],Sort_],dtype='float32')
for j_ in range(labels_te.shape[0]):
    transform=labels_dic[labels_te[j_]]
    One_hot_te[j_,transform]=1


#随机抽取训练数据
np.random.seed(1)#固定初始化
def rand_tr(size_):
    r_index=np.random.randint(0,high=4800,size=size_) 
    n_data=np.zeros([size_,32,32,1],dtype='float32')
    n_labels=np.zeros([size_,Sort_],dtype='float32')
    j=0
    for i in  r_index:
        n_data[j]=data_tr[i]
        n_labels[j]=One_hot_tr[i]
        j+=1
    return n_data,n_labels

##构建tensorflow计算图
import tensorflow as tf

tf.reset_default_graph()#重置计算图
tf.set_random_seed(1)#固定初始化

x=tf.placeholder(tf.float32,[None,32,32,1],name='x_date')
y_=tf.placeholder(tf.float32,[None,Sort_])
#
#卷积
w1=tf.Variable(tf.random_normal([5,5,1,32],stddev=0.01,dtype='float32'))
b1=tf.Variable(tf.random_normal([32],stddev=0.01,dtype='float32'))
conv1=tf.nn.conv2d(x,w1,strides=[1,1,1,1],padding='SAME')
conv1 = tf.nn.relu(conv1+b1)
#池化
pool1=tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#再卷积
w2=tf.Variable(tf.random_normal([5,5,32,64],stddev=0.01,dtype='float32'))
b2=tf.Variable(tf.random_normal([32],stddev=0.01,dtype='float32'))
conv2=tf.nn.conv2d(pool1,w2,strides=[1,1,1,1],padding='SAME')
conv2 = tf.nn.relu(conv2+b2)

#再池化
pool2=tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#密集连接层
pool2_f=tf.reshape(pool2,[-1,8*8*32])#将数据展开为一维数组
w3=tf.Variable(tf.random_normal([8 * 8 * 64,50],stddev=0.01,dtype='float32'))
b3=tf.Variable(tf.random_normal([50],stddev=0.01,dtype='float32'))
y_fc=tf.nn.relu(tf.matmul(pool2_f, w3) + b3)

#Dropout层
keep_prob = tf.placeholder("float32",name='keep_prob')
h_fc1_drop = tf.nn.dropout(y_fc, keep_prob)

#输出层
w4=tf.Variable(tf.random_normal([50,Sort_],stddev=0.01,dtype='float32'))
b4=tf.Variable(tf.random_normal([Sort_],stddev=0.01,dtype='float32'))
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, w4) + b4,name='pre')

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv+1e-8))#损失函数

#设置学习动态
global_step=tf.Variable(0,trainable=False)
learn_rate=tf.train.exponential_decay(1e-4,global_step,100,0.50)
#获取优化器
train=tf.train.AdamOptimizer(learn_rate).minimize(cross_entropy)

saver = tf.train.Saver()#获取存储器
init = tf.global_variables_initializer()#初始所有变量

#构建tensorflow会话
with tf.Session() as sess:
    sess.run(init)
    for i_ in range(500):
        pre=tf.equal(tf.argmax(y_conv,axis=1),tf.argmax(y_,axis=1))#预测值
        acc,cost=sess.run([pre,cross_entropy],feed_dict={x:data_te,y_:One_hot_te,keep_prob: 1.0 } )
        print(i_,'test  acc:%.6f'%(sum(acc)/len(acc)),'  cost:%.6f'%(cost/data_te.shape[0]))
        data_tr_r,One_hot_tr_r=rand_tr(480)
        acc_,cost_,_=sess.run([pre,cross_entropy,train],feed_dict={x:data_tr_r,y_:One_hot_tr_r,keep_prob: 0.8})
        print(i_,'train acc:%.6f'%(sum(acc_)/len(acc_)),'  cost:%.6f'%(cost_/data_tr.shape[0]),'\n')

    saver.save(sess,'E:/python/cnn/model_10/train_model')

