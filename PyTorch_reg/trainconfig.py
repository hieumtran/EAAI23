import torch
from PIL import Image
import numpy as np

class trainConfig:
    def __init__(self, init_model, batch_size, epochs, optimizer, lossFunc):
        self.batch_size = batch_size 
        self.optimizer = optimizer(init_model.parameters())
        if torch.cuda.is_available(): self.device = 'cuda'
        else: self.device = 'cpu'
        self.lossFunc = lossFunc()
        self.epochs = epochs
    
    def normalize(self, input):
        return input / 255
    
    def batch(self, input_paths, output):
        # loop over the dataset
        for i in range(0, len(input_paths), self.batch_size):
            # yield a tuple of the current batched data and labels
            imgs_holder = []
            for batch_path in input_paths[i:i+self.batch_size]:
                imgs_holder.append(self.normalize(np.asarray(Image.open(batch_path))))
            yield (np.stack(imgs_holder, axis=0), output[i:i + self.batch_size])
    

    
    