import csv
# Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# Generate Population Data
df = pd.DataFrame(np.loadtxt(fd_train_mul)) #normal
#df = pd.DataFrame(np.random.randint(0,100,size=(10000,1))) #uniform
pop_mu = df.mean(axis=0)
pop_st = df.std(axis=0)
# Generate Sample Means and Standard Deviations
s_mu = [df.sample(100).mean()[0] for i in range(1000)]
# Plot Sample Means
plt.figure(figsize=(20,10))
sns.distplot(s_mu).grid()
plt.title('Sampling Distribution of Sample Mean (100 samples where N = 1000)')
plt.axvline(x=np.mean(s_mu), label='Mean of Sample Means')
plt.axvline(x=np.mean(s_mu) + np.std(s_mu), label='Std of Sample means', color='r')
plt.axvline(x=np.mean(s_mu) - np.std(s_mu), label='Std of Sample means', color='r')
plt.legend()
plt.show()