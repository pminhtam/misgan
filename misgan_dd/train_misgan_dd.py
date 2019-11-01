from misgan_dd.model_dd import *
from misgan_dd.data_generate import GenerateData
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--data-file",type=str,default = "data/uce_train.csv",help="path to data file")
parser.add_argument("--save-path",type=str,default="./model/uce2",help="path to save model")
parser.add_argument("--epochs",type=int,default=5000,help="epochs ")
parser.add_argument("--batch-size",type=int,default=4096,help="batch size")
parser.add_argument("--n_iter_d",type=int,default=2,help="iter D each batch")
parser.add_argument("--n_iter_g",type=int,default=1,help="iter G each batch")
parser.add_argument("--beta1",type=float,default=0.0001,help="beta1 in Adam optimizer")
parser.add_argument("--beta2",type=float,default=0.0001,help="beta2 in Adam optimizer")
parser.add_argument("--lr",type=float,default=5e-4,help="learning rate")
parser.add_argument("--lambda_",type=int,default=10,help="to calc gradient")
args = parser.parse_args()


def main():
    batch_size = args.batch_size
    epochs = args.epochs
    n_iter_d = args.n_iter_d
    n_iter_g = args.n_iter_g
    beta1 = args.beta1
    beta2 = args.beta2
    lr = args.lr
    lambda_ = args.lambda_


    data_file = args.data_file
    Data = np.genfromtxt(data_file, delimiter=",", filling_values=0)

    data = GenerateData(Data)
    netG_imp,netD_imp = train(data,batch_size,epochs,n_iter_d,n_iter_g,beta1,beta2,lr,lambda_)

    save_path = args.save_path
    torch.save(netG_imp.state_dict(), save_path + "_G.pt")
    torch.save(netD_imp.state_dict(), save_path + "_D.pt")


if __name__ == "__main__":
    main()