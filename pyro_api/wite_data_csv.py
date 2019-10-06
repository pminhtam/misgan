import numpy as np
import csv


data_file = "fd-reduced-30.csv"
Data = np.loadtxt(data_file, delimiter=",",skiprows=1,usecols = (3, 4, 5, 6, 7,14,15,16))[-50000:,:]
# print(np.array([Data[:,0]+Data[:,1]]).T)
Data = np.append(Data,np.array([Data[:,0]+Data[:,1]]).T,1)
Data = np.append(Data,np.array([Data[:,2]+Data[:,3]]).T,1)
Data = np.append(Data,np.array([Data[:,0]*Data[:,1]]).T,1)
Data = np.append(Data,np.array([Data[:,2]*Data[:,3]]).T,1)

with open('fd.csv', 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL,delimiter=',')
    wr.writerow(['A','B','C','D','E','F','G','H','I','J','K','L'])
    for i in Data:
      wr.writerow(i)