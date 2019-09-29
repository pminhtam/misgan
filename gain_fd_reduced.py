import cf
import numpy as np
import tensorflow as tf
from data_gain import Data_Generate
from model_gain import *


mb_size = 256
p_miss = 0.2
p_hint = 0.9
alpha = 10
train_rate = 1.0


data_file = "fd-reduced-30.csv"
Data = np.loadtxt(data_file, delimiter=",",skiprows=1,usecols = (3, 4, 5, 6, 7))[:cf.num_row,:]

Dim,Train_No,trainX,trainM = Data_Generate(Data)

X,M,H,New_X,D_loss1,G_loss1,MSE_train_loss,MSE_test_loss,D_solver,G_solver,G_sample = make_model(Dim)

# Sessions
sess = tf.Session()
sess.run(tf.global_variables_initializer())
np.random.seed(0)
np.random.shuffle(trainX)
train_loss_curr,test_loss_curr = train(X,M,H,New_X,D_loss1,G_loss1,MSE_train_loss,MSE_test_loss,D_solver,G_solver,G_sample,Dim,Train_No,trainX,trainM,sess)

saver = tf.train.Saver()
saver.save(sess, "./model/gain_fd_reduced.ckpt")

import matplotlib.pyplot as plt
plt.subplot(2, 1, 1)
plt.plot(train_loss_curr)
plt.subplot(2, 1, 2)
plt.plot(test_loss_curr)
plt.savefig("./img/"+ data_file + "_gain.jpg")