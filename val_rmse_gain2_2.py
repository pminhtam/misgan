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
Data = np.append(Data,np.array([Data[:,0]+Data[:,1]]).T,1)
Data = np.append(Data,np.array([Data[:,2]+Data[:,3]]).T,1)
Data = np.append(Data,np.array([Data[:,0]*Data[:,1]]).T,1)
Data = np.append(Data,np.array([Data[:,2]*Data[:,3]]).T,1)

print(Data.shape)

Dim,Train_No,trainX,trainM = Data_Generate(Data)

X,M,H,New_X,D_loss1,G_loss1,MSE_train_loss,MSE_test_loss,D_solver,G_solver,G_sample = make_model(Dim)

# Sessions
sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
saver.restore(sess, "./model/gain_h2.ckpt")



Missing0 = np.ones((Train_No,Dim))

Missing0[:,2] = 0
Missing0[:,3] = 0
Missing0[:,4] = 0
Missing0[:,5] = 0
Missing0[:,6] = 0
Missing0[:,7] = 0
Missing0[:,8] = 0
Missing0[:,9] = 0
Missing0[:,10] = 0
# Missing0[:,11] = 0
# Missing0[:,12] = 0
# Missing0[:,13] = 0
# Missing0[:,14] = 0
# Missing0[:,15] = 0

Z_mb = sample_Z(Train_No, Dim)
M_mb = Missing0
X_mb = trainX
New_X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb  # Missing Data Introduce

MSE_final,G_sample = sess.run([MSE_test_loss,G_sample], feed_dict={X: trainX, M: M_mb, New_X: New_X_mb})
print(np.mean(np.square(np.subtract(G_sample*(1-M_mb),trainX*(1-M_mb))),axis=0))
