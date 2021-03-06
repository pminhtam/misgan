import tensorflow as tf
import numpy as np
import cf

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


def discriminator(new_x, h,D_W1,D_W2,D_W3,D_b1,D_b2,D_b3):
    inputs = tf.concat(axis=1, values=[new_x, h])  # Hint + Data Concatenate
    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
    D_logit = tf.matmul(D_h2, D_W3) + D_b3
    D_prob = tf.nn.sigmoid(D_logit)  # [0,1] Probability Output

    return D_prob

# Random sample generator for Z
def sample_Z(m, n):
    return np.random.uniform(0., 0.01, size = [m, n])

# # Mini-batch generation
def sample_idx(m, n):
    A = np.random.permutation(m)
    idx = A[:n]
    return idx


def make_model(Dim,alpha = 10):
    H_Dim1 = Dim
    H_Dim2 = Dim
    # 1.1. Data Vector
    X = tf.placeholder(tf.float32, shape=[None, Dim])
    # 1.2. Mask Vector
    M = tf.placeholder(tf.float32, shape=[None, Dim])
    # 1.3. Hint vector
    H = tf.placeholder(tf.float32, shape=[None, Dim])
    B = tf.placeholder(tf.float32, shape=[None, Dim])
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
    Hat_New_X = New_X * M + G_sample * (1 - M)

    # Discriminator
    D_prob = discriminator(Hat_New_X, H, D_W1, D_W2, D_W3, D_b1, D_b2, D_b3)

    D_loss1 = -tf.reduce_mean(tf.multiply(M,B) * tf.log(D_prob + 1e-8) + tf.multiply((1-M),B) * tf.log(1. - D_prob + 1e-8))
    G_loss1 = -tf.reduce_mean(tf.multiply((1-M),B) * tf.log(D_prob + 1e-8))
    MSE_train_loss = tf.reduce_mean((M * New_X - M * G_sample) ** 2) / tf.reduce_mean(M)

    D_loss = D_loss1
    G_loss = G_loss1 + alpha * MSE_train_loss

    # %% MSE Performance metric
    MSE_test_loss = tf.reduce_mean(((1 - M) * X - (1 - M) * G_sample) ** 2) / tf.reduce_mean(1 - M)

    D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
    G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)
    return X,M,H,B,New_X,D_loss1,G_loss1,MSE_train_loss,MSE_test_loss,D_solver,G_solver,G_sample

def train(X,M,H,B,New_X,D_loss1,G_loss1,MSE_train_loss,MSE_test_loss,D_solver,G_solver,G_sample,Dim,Train_No,trainX,trainM,sess):
    # %% Start Iterations
    mb_size = 1024
    p_hint = 0.8

    train_loss_curr = []
    test_loss_curr = []
    for it in range(cf.gain_iter):
        # %% Inputs
        # print(it)
        for ii in range(int(Train_No / mb_size)):
            mb_idx = [i for i in range(ii * mb_size, (ii + 1) * mb_size)]
            # mb_idx = sample_idx(Train_No, mb_size)
            X_mb = trainX[mb_idx, :]        # data ban dau

            Z_mb = sample_Z(mb_size, Dim)   # random matrix
            M_mb = trainM[mb_idx, :]        # mask matrix
            H_mb1 = sample_M(mb_size, Dim, 1 - p_hint)  # B
            H_mb = (M_mb * H_mb1) + 0.5*(1-H_mb1)   # hint matrix

            New_X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb  # Missing Data Introduce

            _, D_loss_curr = sess.run([D_solver, D_loss1], feed_dict={M: M_mb, New_X: New_X_mb, H: H_mb,B:H_mb1})
            _, G_loss_curr, MSE_train_loss_curr, MSE_test_loss_curr = sess.run(
                [G_solver, G_loss1, MSE_train_loss, MSE_test_loss],
                feed_dict={X: X_mb, M: M_mb, New_X: New_X_mb, H: H_mb,B:H_mb1})
            train_loss_curr.append(MSE_train_loss_curr)
            test_loss_curr.append(MSE_test_loss_curr)

    return train_loss_curr,test_loss_curr