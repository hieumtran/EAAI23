import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
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
    

def sample_func(subset, n_sample, seed):
    # Downsample method
    num_class = set(train['class'])
    subset = []
    for i in num_class:
        subset_class = train.loc[train['class'] == i]
        if len(subset_class[:]) < n_sample:
            subset_class = subset_class.sample(n=n_sample, replace=True, random_state=seed)
        subset.append(subset_class[:n_sample])
    subset = pd.concat(subset)
    return subset

def split_train(subset, seed, save_name, save_loc):
    subset = subset.sample(frac=1, random_state=seed).reset_index(drop=True)
    subset.to_csv(save_loc+save_name+'.csv', header=None, index=False)
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

def viz(subset, colormap):
    emotions = [i for i in range(8)]
    labels = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt']
    cmap = plt.get_cmap(colormap)
    colors = cmap(np.linspace(0, 1, len(emotions)))
    plt.figure(figsize=(15,15))

    for i in range(len(emotions)):
        emotion_subset = subset.loc[subset['class'] == emotions[i]]
        plt.scatter(emotion_subset['val'], emotion_subset['ars'], color=colors[i], label=labels[i], zorder=2)
    
    # Valence and Arousal Axis
    plt.axhline(y=0, xmin=0, xmax=0.93, linewidth=8, color='black', zorder=1)
    plt.axvline(x=0, ymin=0, ymax=0.93, linewidth=8, color='black', zorder=1)
    plt.arrow(1.1, 0, 0.01, 0, width=0.013, color='black')
    plt.arrow(0, 1.1, 0, 0, width=0.013, color='black')

    # Text for annotation purpose
    plt.text(0.1, 1.1, 'Valence', fontsize='25', fontweight='bold')
    plt.text(1.1, -0.1, 'Arousal', fontsize='25', fontweight='bold')

    plt.tight_layout()
    plt.axis('off') #hide axes and borders
    plt.legend(markerscale=3., fontsize='25')
    plt.savefig('./PyTorch_reg/figure/russell_affectnet.jpg', dpi=500)
    
    

if __name__ == "__main__":
    # save_all_train()

    # Init train
    train = pd.read_csv('./data/train.csv')
    # # Plot figure
    # # viz(train, 'Dark2')


    # # Augment + Downsampling
    # # # Apply downsampling
    n_sample = 25000
    seed = 1
    subset = sample_func(train, n_sample, seed)
    # print(subset['class'].value_counts())

    # # # Augmentation
    # HF_4 = augmentation(subset, 4, 'HF')
    # CJ_5 = augmentation(subset, 5, 'CJ')
    # HF_5 = augmentation(subset, 5, 'HF')
    # CJ_7 = augmentation(subset, 7, 'CJ')
    # HF_7 = augmentation(subset, 7, 'HF')
    # # Concat all augmentation
    # subset = pd.concat([subset, HF_4, CJ_5, HF_5, CJ_7, HF_7]).reset_index(drop=True)
    # # Split train
    split_train(subset, seed, 'train_resample_25000', './data/')
    tmp = pd.read_csv('./data/train_resample_25000.csv')
    print(tmp)