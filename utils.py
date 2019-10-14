import numpy as np

def get_error_entropy(data, vt, vp):
    def entropy(data):
        tuples, counts = np.unique(data, axis=0, return_counts=True)
        ps = counts/data.shape[0]
        ent = np.sum(-ps*np.log2(ps))
        return ent

    def fxy(x, y):
        xy = np.zeros((x.shape[0], x.shape[1]+1))
        xy[:,:x.shape[1]] = x
        xy[:,-1] = y
        return (entropy(x) + entropy(y) - entropy(xy)) / entropy(y)

    def error(x,y):
        return 1 - fxy(x,y)
    
    return error(data[:,vt], data[:,vp])
    