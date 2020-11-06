import tensorflow as tf
import numpy as np
import cf
from tqdm import tqdm

# 1. Xavier Initialization Definition
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


# Hint Vector Generation
def sample_M(m, n, p):
    A = np.random.uniform(0., 1., size=[m, n])
    B = A > p
    C = 1. * B
    return C


def generator(new_x, m,G_W1,G_W2,G_W3,G_b1,G_b2,G_b3):
    inputs = tf.concat(axis=1, values=[new_x, m])  # Mask + Data Concatenate
    G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
    G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
    G_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3)  # [0,1] normalized Output

    return G_prob

# Random sample generator for Z
def sample_Z(m, n):
    return np.random.uniform(0., 0.01, size = [m, n])
    # return np.random.uniform(0., 0.5, size = [m, n])

# # Mini-batch generation
def sample_idx(m, n):
    A = np.random.permutation(m)
    idx = A[:n]
    return idx


def make_model(Dim):
    H_Dim1 = Dim
    H_Dim2 = Dim
    # 1.1. Data Vector
    X = tf.placeholder(tf.float32, shape=[None, Dim])
    # 1.2. Mask Vector
    M = tf.placeholder(tf.float32, shape=[None, Dim])
    # 1.3. Hint vector
    H = tf.placeholder(tf.float32, shape=[None, Dim])
    # 1.4. X with missing values
    New_X = tf.placeholder(tf.float32, shape=[None, Dim])

    D_W1 = tf.Variable(xavier_init([Dim * 2, H_Dim1]))  # Data + Hint as inputs
    D_b1 = tf.Variable(tf.zeros(shape=[H_Dim1]))

    D_W2 = tf.Variable(xavier_init([H_Dim1, H_Dim2]))
    D_b2 = tf.Variable(tf.zeros(shape=[H_Dim2]))

    D_W3 = tf.Variable(xavier_init([H_Dim2, Dim]))
    D_b3 = tf.Variable(tf.zeros(shape=[Dim]))  # Output is multi-variate

    theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]

    G_W1 = tf.Variable(
        xavier_init([Dim * 2, H_Dim1]))  # Data + Mask as inputs (Random Noises are in Missing Components)
    G_b1 = tf.Variable(tf.zeros(shape=[H_Dim1]))

    G_W2 = tf.Variable(xavier_init([H_Dim1, H_Dim2]))
    G_b2 = tf.Variable(tf.zeros(shape=[H_Dim2]))

    G_W3 = tf.Variable(xavier_init([H_Dim2, Dim]))
    G_b3 = tf.Variable(tf.zeros(shape=[Dim]))

    theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]

    # Generator
    G_sample = generator(New_X, M, G_W1, G_W2, G_W3, G_b1, G_b2, G_b3)

    # Combine with original data
    MSE_train_loss = tf.reduce_mean((New_X - G_sample) ** 2)

    G_loss = MSE_train_loss

    # %% MSE Performance metric
    MSE_test_loss = tf.reduce_mean(((1 - M) * X - (1 - M) * G_sample) ** 2) / tf.reduce_mean(1 - M)
    MSE_loss_cols = tf.reduce_mean(((1 - M) * X - (1 - M) * G_sample) ** 2, axis=0) / tf.reduce_mean(1 - M, axis=0)

    G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)
    return X,M,H,New_X,MSE_train_loss,MSE_test_loss,G_solver,G_sample, MSE_loss_cols


def train(X,M,H,New_X,MSE_train_loss,MSE_test_loss,G_solver,G_sample,Dim,Train_No,trainX,p_miss,sess,gain_iter,batch_size,p_hint,prob_masks):
    # import tensorboardX
    # import torch
    p_miss = 1/Dim 
    print("Init p_miss = ", p_miss)
    # %% Start Iterations
    mb_size = batch_size
    p_hint = p_hint

    train_loss_curr = []
    test_loss_curr = []
    # writer = tensorboardX.SummaryWriter()
    prob_cols_tracks = [[] for _ in range(Dim)]
    total_iter = 0

    threshold_mean = 0.1 
    threshold_std = 0.02
    max_counter = 200

    pmisses = [p_miss for _ in range(Dim)]
    vp_cols = [0 for _ in range(Dim)] # -1 = vt, 0 = unknown, 1 = vp

    for it in tqdm(range(gain_iter)):
        # %% Inputs
        # print(it)
        np.random.shuffle(trainX)
        
        for ii in range(int(Train_No / mb_size)):
            mb_idx = [i for i in range(ii * mb_size, (ii + 1) * mb_size)]
            # mb_idx = sample_idx(Train_No, mb_size)
            X_mb = trainX[mb_idx, :]

            Z_mb = sample_Z(mb_size, Dim)
            # M_mb = trainM[mb_idx, :]
            #M_mb = np.zeros_like(X_mb)
            M_mb = np.ones_like(X_mb)

            ## FD only
            #for i in range(Dim):       # all column
            for i in [2]:               # some column
                ## FD mask random
                M_mb[:,i] = np.random.choice(2, size=(X_mb.shape[0],), p=[p_miss, 1-p_miss])
                ## FD mask total
                # M_mb[:,i] = np.zeros((X_mb.shape[0],))

            
            #print(M_mb)
            H_mb1 = sample_M(mb_size, Dim, 1 - p_hint)
            H_mb = M_mb * H_mb1

            New_X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb  # Missing Data Introduce

            _,  MSE_train_loss_curr, MSE_test_loss_curr, prob_masks_vals = sess.run(
                [G_solver, MSE_train_loss, MSE_test_loss, prob_masks],
                feed_dict={X: X_mb, M: M_mb, New_X: New_X_mb, H: H_mb})
            train_loss_curr.append(MSE_train_loss_curr)
            test_loss_curr.append(MSE_test_loss_curr)


            for i in range(int(Dim)):
                prob_cols_tracks[i].append(prob_masks_vals[i])
            
        if (it+1) % (gain_iter//2) == 0:
            for i in range(int(Dim)): 
                mean_prob = np.mean(prob_cols_tracks[i][-max_counter:])
                std_prob = np.std(prob_cols_tracks[i][-max_counter:])
                print("col {}: mean: {:.3f} - std: {:.3f}".format(i, mean_prob, std_prob))

                if mean_prob < threshold_mean and std_prob < threshold_std:
                    # col i is VP
                    vp_cols[i] = 1
                    pmisses[i] = 1 - p_miss
                else:
                    vp_cols[i] = -1
                    pmisses[i] /= 4
    
        total_iter += 1

        # for i in range(int(Dim)):
        #     writer.add_scalar("prob_col/{}".format(i), torch.FloatTensor([prob_masks_vals[i]]), it)

            # total_iter += 1
    return train_loss_curr,test_loss_curr
