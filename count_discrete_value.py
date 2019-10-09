from gain.model_gain import *


data_file = "data/lineitem.tbl.8"
Data = np.loadtxt(data_file, delimiter="|",skiprows=1,usecols = (0, 1, 2, 4, 5))[:cf.num_row,:]
value = {}
for i in Data[:,2]:
    if i in value.keys():
        value[i]+=1
    else:
        value[i] = 1
print(value)
print(len(value.keys()))