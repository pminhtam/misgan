from gain.data_gain import Data_Generate
from gain.model_gain import *
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

parser = argparse.ArgumentParser()

parser.add_argument("--data-file",type=str,default = "data/uce_train.csv",help="path to data file")
parser.add_argument("--save-path",type=str,default="./model/gain_uce2.ckpt",help="path to save model")
parser.add_argument("--iter",type=int,default=8000,help="iter")
parser.add_argument("--batch-size",type=int,default=4096,help="batch size")
parser.add_argument("--p-miss",type=float,default=0.2,help="p miss")
parser.add_argument("--p-hint",type=float,default=0.9,help="p hint")
parser.add_argument("--alpha",type=int,default=10,help="alpha")
args = parser.parse_args()


def main():
    p_miss = args.p_miss
    p_hint = args.p_hint
    alpha = args.alpha
    gain_iter = args.iter
    batch_size = args.batch_size

    data_file = args.data_file
    Data = np.genfromtxt(data_file, delimiter=",", filling_values=0)
    Dim,Train_No,trainX,trainM = Data_Generate(Data,p_miss)
    X,M,H,New_X,D_loss1,G_loss1,MSE_train_loss,MSE_test_loss,D_solver,G_solver,G_sample, prob_masks = make_model(Dim,alpha)

    # Sessions
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)

    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    # sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    np.random.seed(cf.gain_seed)
    np.random.shuffle(trainX)
    train_loss_curr,test_loss_curr = train(X,M,H,New_X,D_loss1,G_loss1,MSE_train_loss,MSE_test_loss,D_solver,G_solver,G_sample,Dim,Train_No,trainX,p_miss,sess,gain_iter,batch_size,p_hint,prob_masks)

    saver = tf.train.Saver()
    saver.save(sess, args.save_path)
if __name__ == "__main__":
    main()