import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import time

def augmentation(subset, emotion, method):
    assert method in ['GB', 'HF', 'CJ']
    emotion_subset = subset.loc[subset['class'] == emotion]
    aug_subset = {'image': [], 'class': [], 'val': [], 'ars': []}
    if method == 'GB':
        transform = transforms.GaussianBlur(kernel_size=(7, 13), sigma=(2, 5))
    if method == 'HF':
        transform = transforms.RandomHorizontalFlip(p=1)
    if method == 'CJ':
        transform = transforms.ColorJitter(brightness=(0.5,1.5), contrast=(1), saturation=(0.5,1.5), hue=(-0.1,0.1))

    start_time = time.time()
    for i in range(len(emotion_subset)):
        save_path = './data/train_set/images/'
        aug_img_name = emotion_subset['image'].iloc[i].replace('.jpg', '') + '_' + method + '.jpg'
        img = Image.open(save_path+emotion_subset['image'].iloc[i])
        
        transform(img).save(save_path+aug_img_name)
        aug_subset['image'].append(aug_img_name)
        aug_subset['class'].append(emotion_subset['class'].iloc[i])
        aug_subset['val'].append(emotion_subset['val'].iloc[i])
        aug_subset['ars'].append(emotion_subset['ars'].iloc[i])

    print('Finish method ' + method + ' on emotion no.' + str(emotion) + ' after ' + str(time.time() - start_time) + ' seconds.')
    return pd.DataFrame(aug_subset)
    

def downsample_func(subset, downsample):
    # Downsample method
    num_class = set(train['class'])
    subset = []
    for i in num_class:
        subset_class = train.loc[train['class'] == i]
        subset.append(subset_class[:downsample])
    subset = pd.concat(subset)
    # print('Number of each classes:')
    # print(subset['class'].value_counts())
    return subset

def split_train(subset, seed, save_name, save_loc):
    subset = subset.sample(frac=1, random_state=seed).reset_index(drop=True)
    # Splitting to regression and classification
    subset_reg = subset.loc[:, ['image', 'val', 'ars']]
    subset_class = subset.loc[:, ['image', 'class']]
    # Saving to csv
    subset_reg.to_csv(save_loc+save_name+'_reg.csv', header=None, index=False)
    subset_class.to_csv(save_loc+save_name+'_class.csv', header=None, index=False)

def save_all_train():
    # Load and Merge Train
    train_reg = pd.read_csv('./data/train_reg.csv', header=None)
    train_class = pd.read_csv('./data/train_class.csv', header=None)
    train = pd.merge(train_class, train_reg, on=0).rename(columns={0: 'image', '1_x': 'class', '1_y': 'val', 2: 'ars'})

    # Load and Merge Val
    val_reg = pd.read_csv('./data/val_reg.csv', header=None)
    val_class = pd.read_csv('./data/val_class.csv', header=None)
    val = pd.merge(val_class, val_reg, on=0).rename(columns={0: 'image', '1_x': 'class', '1_y': 'val', 2: 'ars'})

    # Concat train and Val
    train = pd.concat([train, val]).reset_index(drop=True)
    # Save to csv
    train.to_csv('./data/train.csv', index=False)

def viz(train):

if __name__ == "__main__":
    # save_all_train()

    # Init train
    train = pd.read_csv('./data/train.csv')



    # # Apply downsampling
    # downsample = 20000
    # subset = downsample_func(train, downsample)

    # # Augmentation
    # HF_4 = augmentation(subset, 4, 'HF')
    # CJ_5 = augmentation(subset, 5, 'CJ')
    # HF_5 = augmentation(subset, 5, 'HF')
    # CJ_7 = augmentation(subset, 7, 'CJ')
    # HF_7 = augmentation(subset, 7, 'HF')
    # # Concat all augmentation
    # subset = pd.concat([subset, HF_4, CJ_5, HF_5, CJ_7, HF_7]).reset_index(drop=True)
    # # Split train
    # split_train(subset, 1, 'train_CJ_HF_20000_', './data/')
    

