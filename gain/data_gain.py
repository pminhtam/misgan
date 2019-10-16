import numpy as np
# Data generation
def Data_Generate(Data ,p_miss = 0.2):
    No = len(Data)
    Dim = len(Data[0,:])

    # Normalization (0 to 1)
    data_max  = np.max(np.abs(Data),axis = 0)
    data_min = np.min(np.abs(Data),axis = 0)
    print(data_max)
    Data = (Data-data_min) / (data_max-data_min + 1e-10)

    p_miss_vec = p_miss * np.ones((Dim, 1))

    Missing = np.zeros((No, Dim))

    for i in range(Dim):
        A = np.random.uniform(0., 1., size=[len(Data), ])
        B = A > p_miss_vec[i]
        Missing[:, i] = 1. * B
    idx = np.random.permutation(No)

    Train_No =  No
    trainX = Data
    trainM = Missing
    return Dim,Train_No,trainX,trainM