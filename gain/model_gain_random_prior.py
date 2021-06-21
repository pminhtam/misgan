import tensorflow as tf
import numpy as np
import cf
from tqdm import tqdm
from gain.model_gain import *

def train(X,M,H,New_X,D_loss1,G_loss1,MSE_train_loss,MSE_test_loss,D_solver,G_solver,G_sample,Dim,Train_No,trainX,p_miss,sess,gain_iter,batch_size,p_hint,
    prob_masks,maskss=[0,1,2]):
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

            Z_mb = np.zeros([mb_size, Dim])
            # M_mb = trainM[mb_idx, :]
            #M_mb = np.zeros_like(X_mb)
            M_mb = np.ones_like(X_mb)

            ## FD only
            #for i in range(Dim):       # all column
            for i in maskss:               # some column
                ## FD mask random
                M_mb[:,i] = np.random.choice(2, size=(X_mb.shape[0],), p=[p_miss, 1-p_miss])
                ## FD mask total
                # M_mb[:,i] = np.zeros((X_mb.shape[0],))

            ## N value mask random
            #miss_ori = np.ones(Dim)
            #miss_pos = np.random.choice(Dim,2,replace=False)
            #miss_pos = [0,1]
            #for ii_pos in miss_pos:
            #    miss_ori[ii_pos] = 0
            #print(miss_ori)
            #for i in range(mb_size):
            #    #M_mb[i,:] = np.random.permutation(miss_ori)
            #    M_mb[i,:] = miss_ori
            
            #print(M_mb)
            H_mb1 = sample_M(mb_size, Dim, 1 - p_hint)
            H_mb = M_mb * H_mb1

            New_X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb  # Missing Data Introduce

            _, D_loss_curr = sess.run([D_solver, D_loss1], feed_dict={M: M_mb, New_X: New_X_mb, H: H_mb})
            _, G_loss_curr, MSE_train_loss_curr, MSE_test_loss_curr, prob_masks_vals = sess.run(
                [G_solver, G_loss1, MSE_train_loss, MSE_test_loss, prob_masks],
                feed_dict={X: X_mb, M: M_mb, New_X: New_X_mb, H: H_mb})
            train_loss_curr.append(MSE_train_loss_curr)
            test_loss_curr.append(MSE_test_loss_curr)

            # for i in range(int(Dim)):
            #     if prob_masks_vals[i] > prob_cols_tracks[i][-1]:
            #         counter_cols[i] += 1
            #         if counter_cols[i] >= max_counter:
            #             # converge -> check mean and std
            #             mean_prob = np.mean(prob_cols_tracks[i][-max_counter:])
            #             std_prob = np.std(prob_cols_tracks[i][-max_counter:])
            #             print("col {}: mean: {:.3f} - std: {:.3f}".format(i, mean_prob, std_prob))
            #     else:
            #         counter_cols[i] = 0

            #     prob_cols_tracks[i].append(prob_masks_vals[i])

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
