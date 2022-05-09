from email import header
from operator import index
import pandas as pd

# Data loading
path = './data/'

train_reg = pd.read_csv('./data/train_reg.csv', header=None)
train_class = pd.read_csv('./data/train_class.csv', header=None)
train = pd.merge(train_class, train_reg, on=0).rename(columns={0: 'image', '1_x': 'class', '1_y': 'val', 2: 'ars'})

val_reg = pd.read_csv('./data/val_reg.csv', header=None)
val_class = pd.read_csv('./data/val_class.csv', header=None)
val = pd.merge(val_class, val_reg, on=0).rename(columns={0: 'image', '1_x': 'class', '1_y': 'val', 2: 'ars'})

train = pd.concat([train, val]).reset_index(drop=True)
# print(train['class'].value_counts())

# downsample = min(train['class'].value_counts())
downsample = 10000

num_class = set(train['class'])

subset = []
for i in num_class:
    subset_class = train.loc[train['class'] == i]
    if len(subset_class) < downsample:
        subset_class = subset_class.sample(n = downsample, replace=True)
        subset.append(subset_class)
    else:
        subset.append(subset_class[:downsample])


subset = pd.concat(subset).sample(frac=1).reset_index(drop=True)
print(subset)
subset_reg = subset.loc[:, ['image', 'val', 'ars']]
subset_class = subset.loc[:, ['image', 'class']]
subset_reg.to_csv('./data/train_subset_resampling_10000_reg.csv', header=None, index=False)
subset_class.to_csv('./data/train_subset_resampling_10000_class.csv', header=None, index=False)