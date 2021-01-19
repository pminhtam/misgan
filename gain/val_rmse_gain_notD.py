from gain.data_gain import Data_Generate
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from gain.model_gain_notD import *
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--data-file",type=str,default = "../gen_data/data_gamma_1_3.csv",help="path to data file")
parser.add_argument("--save-path",type=str,default="./model/data_gamma_1_3.ckpt",help="path to save model")
parser.add_argument("--mask",type=str,default="2",help="mask")
parser.add_argument("--remain",type=str,default="",help="mask")
args = parser.parse_args()


def main():
    np.random.seed(0)
    data_file = args.data_file
    mask = args.mask
    remain = [int(i) for i in args.remain]
    mask = [int(i) for i in args.mask.split(',')]
    Data = np.genfromtxt(data_file, delimiter=",", filling_values=0)

    Dim,Train_No,trainX,trainM = Data_Generate(Data)

    X,M,H,New_X,MSE_train_loss,MSE_test_loss,G_solver,G_sample, MSE_loss_cols = make_model(Dim)

    # Sessions
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    saver.restore(sess, args.save_path)



    Missing0 = np.ones((Train_No,Dim))
    # print(mask.split(','))
    A = [0,1,2,3,4,5]
    if len(remain) > 0:
        mask=list(set(A) - set(remain))
    for i in mask:
        Missing0[:,int(i)] = 0


    Z_mb = sample_Z(Train_No, Dim)
    M_mb = Missing0
    X_mb = trainX
    New_X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb  # Missing Data Introduce

    MSE_final,G_sample = sess.run([MSE_test_loss,G_sample], feed_dict={X: trainX, M: M_mb, New_X: New_X_mb})

    #print("Re gen missing value  :",*np.mean(np.square(np.subtract(G_sample*(1-M_mb),trainX*(1-M_mb))),axis=0))
    #print("Re gen missing value  :",*np.mean(np.square(np.subtract(G_sample*M_mb,trainX*M_mb)),axis=0))
    print("Re gen all  :",np.mean(np.square(np.subtract(G_sample,trainX)),axis=0))
    re_all = np.mean(np.square(np.subtract(G_sample,trainX)),axis=0)
    with open("log_6col.txt","a+") as f:
        # f.writelines(str(re_all[2])+ "|" + str(re_all[5])+"\n")
        f.writelines(str(re_all[2]) + "\n")
    #print(np.sqrt(np.mean(np.square(np.subtract(G_sample[2],G_sample[1]+G_sample[0])))))
    s = 0
    #for i in range(len(G_sample)):
        # print(G_sample[i])
        # print(G_sample[i][0], "   ",G_sample[i][1]*17, "   ", G_sample[i][2]*18 , "   " ,G_sample[i][2]*18 - G_sample[i][1]*17 -G_sample[i][0] )
        # s += np.square(G_sample[i][5]*4 - G_sample[i][4]*2 *G_sample[i][3]*2)
        #s += np.square(G_sample[i][1] - G_sample[i][0])
    #print(np.sqrt(s / (len(trainX))))


if __name__ == "__main__":
    main()
