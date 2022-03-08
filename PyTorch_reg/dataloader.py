import torch
import numpy as np

def path_loader(dtype, outtype):
    assert dtype in ['train', 'test', 'val']
    assert outtype in ['reg', 'class']

    if dtype in ['train', 'val']: imgs_path = './train_set/images/'
    elif dtype == 'test': imgs_path = './val_set/images/'

    imgs_labels = np.load('./data/' + dtype + '_path.npy') # Shuffle imgs
    out_labels = np.load('./data/' + dtype + '_' + outtype + '.npy') # labels: class or reg

    # Load images data to numpy array
    input_paths = [imgs_path + img_label + '.jpg' for img_label in imgs_labels]
    return input_paths, out_labels.astype(float)