import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from torchvision import transforms


# inherit the torch.utils.data.Dataset class
class Dataset(Dataset):
    def __init__(self, image_dir, label_frame, mode, subset=None, transform=None):
        """
        Args:
            image_dir (string): Directory with all the images.
            label_frame (string): Path to the csv file with class or regression labels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image_dir = image_dir
        self.label_frame = pd.read_csv(label_frame)
        assert mode in ['reg', 'class', 'both']
        self.mode = mode
        self.transform = transform
        if subset != None: self.label_frame = self.label_frame[:subset]
        
    def __len__(self):
        return self.label_frame.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.image_dir,
                                self.label_frame.iloc[idx, 0])
        images = Image.open(img_name)

        if (self.transform):
            images = self.transform(images)
        
        if self.mode == 'reg':
            labels = self.label_frame.iloc[idx, 1:]
            labels = np.array([labels]).astype('float').reshape(-1, 2)
            return images, labels
        elif self.mode == 'class':
            labels = self.label_frame.iloc[idx, 1]
            labels = (np.array(labels)).astype("int")
            return images, labels
        # elif self.mode == 'both':
        #     labels_class = self.label_frame.iloc[idx, 1]
        #     labels_class = (np.array(labels_class)).astype("int")
        #     labels_reg = self.label_frame.iloc[idx, 2:]
        #     labels_reg = np.array([labels_reg]).astype('float').reshape(-1, 2)
        #     return images, labels_reg, labels_class

        return images, labels


class Dataloader():
    def __init__(
        self, root, image_dir, label_frame,
        transform=transforms.Compose([transforms.ToTensor(),
                transforms.Normalize(mean=[0.5686, 0.4505, 0.3990],std=[0.2332, 0.2064, 0.1956])]),
        mode=None, subset = None
    ):
        """
        Args:
            root (str): directory to the EAAI23 folder (the folder \
                containing the folder data/train_set/images (or \
                val_set/images for testing))
            image_dir (str): directory from EAAI23 to the images folder.
            label_frame (str): the .csv file in the data folder corresponding \
                to the images in the image folder provided in the image_dir.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root = root
        self.image_dir = root + image_dir
        self.label_frame = root + "data/" + label_frame
        self.mode = mode
        self.transform = transform
        self.subset = subset

    def load_batch(self, batch_size, shuffle=False, num_workers=0):
        """
        Args:
            batch_size (int): number of data points in a batch
            shuffle (bool, optional): shuffle the dataset before splitting into batch
            num_workers (int, optional): how many subprocesses to use for data 

        Returns:
            torch.utils.data.DataLoader object. Looping through the DataLoader object \
                will return the corresponding batchX, batchY.
        """
        
        dataset = Dataset(self.image_dir, self.label_frame, subset=self.subset, transform=self.transform, mode=self.mode)
        batch_iter = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

        return batch_iter
