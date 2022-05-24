import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# Data loading
path = './data/'

train_reg = pd.read_csv('./data/train_subset_reg.csv', header=None)
train_class = pd.read_csv('./data/train_subset_class.csv', header=None)
train = pd.merge(train_class, train_reg, on=0).rename(columns={0: 'image', '1_x': 'class', '1_y': 'val', 2: 'ars'})

# val_reg = pd.read_csv('./data/val_reg.csv', header=None)
# val_class = pd.read_csv('./data/val_class.csv', header=None)
# val = pd.merge(val_class, val_reg, on=0).rename(columns={0: 'image', '1_x': 'class', '1_y': 'val', 2: 'ars'})

# train = pd.concat([train, val]).reset_index(drop=True)
print(train)
for i in range(0, 8):
    plt.scatter(train.loc[train['class'] == i]['val'], train.loc[train['class'] == i]['ars'])
    plt.xticks(np.arange(-1, 1.2, 0.2))
    plt.yticks(np.arange(-1, 1.2, 0.2))
    plt.show()