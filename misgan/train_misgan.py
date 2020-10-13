from misgan.model import *
from misgan.data_generate import GenerateData
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--data-file",type=str,default = "../data/data1_train.csv",help="path to data file")
parser.add_argument("--save-path",type=str,default="../model/uce2",help="path to save model")
parser.add_argument("--nz",type=int,default=256,help="size of Generator input vector")
parser.add_argument("--batch-size",type=int,default=4096,help="batch size")
parser.add_argument("--alpha",type=float,default=0.2,help="ratio weight between mask_loss and data_loss")
parser.add_argument("--beta",type=float,default=0.2,help="ratio weight between data_loss and impu_loss")
parser.add_argument("--epoch_1",type=int,default=5000,help="epoch data, mask  train")
parser.add_argument("--epoch_2",type=int,default=5000,help="epoch of data, mask, imputer  train")
parser.add_argument("--lrate1",type=float,default=1e-4,help="learning rate of data, mask  train")
parser.add_argument("--imputer_lrate",type=float,default=2e-4,help="learning rate of data, mask, imputer  train")
args = parser.parse_args()
torch.autograd.set_detect_anomaly(True)

def main():
    batch_size = args.batch_size
    nz = args.nz
    alpha = args.alpha
    beta = args.beta
    epoch_1 = args.epoch_1
    epoch_2 = args.epoch_2
    lrate1 = args.lrate1
    imputer_lrate = args.imputer_lrate


    data_file = args.data_file
    Data = np.genfromtxt(data_file, delimiter=",", filling_values=0)

    data = GenerateData(Data)
    data_gen, mask_gen, imputer, loss = train(data,batch_size,nz,alpha,beta,epoch_1,epoch_2,lrate1,imputer_lrate)

    save_path = args.save_path
    torch.save(data_gen.state_dict(), save_path + "_data.pt")
    torch.save(mask_gen.state_dict(), save_path + "_mask.pt")
    torch.save(imputer.state_dict(), save_path + "_imputer.pt")
    import matplotlib.pyplot as plt
    plt.plot(loss)
    plt.savefig( save_path + "_loss.jpg")

if __name__ == "__main__":
    main()