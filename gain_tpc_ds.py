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


data_file = "store_returns.dat"
Data = np.genfromtxt(data_file, delimiter="|",skip_header=1,usecols = (0,1,2,3,4,5), filling_values=0)[:cf.num_row,:]


Dim,Train_No,trainX,trainM = Data_Generate(Data)

X,M,H,New_X,D_loss1,G_loss1,MSE_train_loss,MSE_test_loss,D_solver,G_solver,G_sample = make_model(Dim)

# Sessions
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# %% Start Iterations
train_loss_curr = []
test_loss_curr = []
for it in range(cf.gain_iter):
    # %% Inputs
    mb_idx = sample_idx(Train_No, mb_size)
    X_mb = trainX[mb_idx, :]

    Z_mb = sample_Z(mb_size, Dim)
    M_mb = trainM[mb_idx, :]
    H_mb1 = sample_M(mb_size, Dim, 1 - p_hint)
    H_mb = M_mb * H_mb1

    New_X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb  # Missing Data Introduce

    _, D_loss_curr = sess.run([D_solver, D_loss1], feed_dict={M: M_mb, New_X: New_X_mb, H: H_mb})
    _, G_loss_curr, MSE_train_loss_curr, MSE_test_loss_curr = sess.run([G_solver, G_loss1, MSE_train_loss, MSE_test_loss],feed_dict={X: X_mb, M: M_mb, New_X: New_X_mb, H: H_mb})
    train_loss_curr.append(MSE_train_loss_curr)
    test_loss_curr.append(MSE_test_loss_curr)
    # %% Intermediate Losses
    if it % 10000 == 0:
        print('Iter: {}'.format(it))
        print('Train_loss: {:.4}'.format(np.sqrt(MSE_train_loss_curr)))
        print('Test_loss: {:.4}'.format(np.sqrt(MSE_test_loss_curr)))
        print()

saver = tf.train.Saver()
saver.save(sess, "./model/gain_ds.ckpt")

import matplotlib.pyplot as plt
plt.subplot(2, 1, 1)
plt.plot(train_loss_curr)
plt.subplot(2, 1, 2)
plt.plot(test_loss_curr)
plt.savefig("./img/"+ data_file + "_gain.jpg")
