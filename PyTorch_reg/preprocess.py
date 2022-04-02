import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial import Delaunay
from scipy.spatial.distance import euclidean
import torch

lnd = np.reshape(np.load('./data/train_set/annotations/0_lnd.npy'), (68, -1))

tri = Delaunay(lnd)
print(tri.simplices)
ma_vertices = torch.zeros((68, 68))
for i in range(len(tri.simplices)):
    
# plt.triplot(lnd[:,0], lnd[:,1], tri.simplices)
# plt.plot(lnd[:,0], lnd[:,1], 'o')
plt.show()