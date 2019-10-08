import cf
import numpy as np
import tensorflow as tf
from data_gain import Data_Generate
from model_gain import *


mb_size = 128
p_miss = 0.2
p_hint = 0.9
alpha = 10
train_rate = 1.0


data_file = "uce-results-by-school-2011-2015.csv"
Data = np.genfromtxt(data_file, delimiter=",",skip_header=1,usecols = (3, 6, 7, 15, 22,14,17,19,21,22,23), filling_values=0)[:cf.num_row,:]
# print(np.array([Data[:,0]+Data[:,1]]).T)
Data = (Data - np.min(np.abs(Data),axis = 0)) / (np.max(np.abs(Data),axis = 0)+ 1e-10 - np.min(np.abs(Data),axis = 0))

Data = np.append(Data,np.array([Data[:,0]+Data[:,1]]).T,1)
Data = np.append(Data,np.array([Data[:,2]+Data[:,3]]).T,1)
Data = np.append(Data,np.array([Data[:,0]*Data[:,1]]).T,1)
Data = np.append(Data,np.array([Data[:,2]*Data[:,3]]).T,1)

Dim,Train_No,trainX,trainM = Data_Generate(Data)

X,M,H,New_X,D_loss1,G_loss1,MSE_train_loss,MSE_test_loss,D_solver,G_solver,G_sample = make_model(Dim)

# Sessions
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)

sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# sess = tf.Session()
sess.run(tf.global_variables_initializer())

np.random.seed(cf.gain_seed)
np.random.shuffle(trainX)
train_loss_curr,test_loss_curr = train(X,M,H,New_X,D_loss1,G_loss1,MSE_train_loss,MSE_test_loss,D_solver,G_solver,G_sample,Dim,Train_No,trainX,trainM,sess)


saver = tf.train.Saver()
saver.save(sess, "./model/gain_uce2.ckpt")

import matplotlib.pyplot as plt
plt.subplot(2, 1, 1)
plt.plot(train_loss_curr)
plt.subplot(2, 1, 2)
plt.plot(test_loss_curr)
plt.savefig("./img/"+ data_file + "_gain.jpg")
