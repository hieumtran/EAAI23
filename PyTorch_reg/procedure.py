import torch
import numpy as np
from PIL import Image
import timeit
import matplotlib.pyplot as plt
import datetime


class procedure:
    def __init__(self, optimizer, loss_func, model, start_epoch, end_epoch, device):
        self.optimizer = optimizer  # optimizer
        self.loss_func = loss_func # loss function
        self.model = model  # model init
        self.device = device  # device
        self.start_epoch = start_epoch # Begining epoch
        self.end_epoch = end_epoch # Ending epoch
        self.train_arr = [] # Init training loss
        self.val_arr = [] # Init validation loss

    def fit(self, train_loader, val_loader):
        # Epoch training
        output_template = 'Epoch {} / train: {:.10f} / val: {:.10f}  / Time: {:.5f}s / Current date & Time: {:%Y-%m-%d %H:%M:%S}'
        for epoch in range(self.start_epoch, self.end_epoch + 1):
            epoch_start = timeit.default_timer()
            train_loss = self.train(train_loader, training=True)
            val_loss = self.train(val_loader, training=False)
            self.train_arr.append(train_loss)
            self.val_arr.append(val_loss)
            print(output_template.format(epoch, train_loss, val_loss, timeit.default_timer() - epoch_start, datetime.datetime.now()))
            
    # Training and Validation
    def train(self, loader, training):
        if training: 
            self.model.train()
            log_template = 'Training mini-batch {}: {:.10f} / Time: {:.5f}s / Current date & Time: {:%Y-%m-%d %H:%M:%S}'
        else: 
            self.model.eval()
            log_template = 'Validation mini-batch {}: {:.10f} / Time: {:.5f}s / Current date & Time: {:%Y-%m-%d %H:%M:%S}'
        
        # Init computation variables
        samples, batch_cnt, sum_loss = 0, 0, 0
        for (batchX, batchY) in loader:
            start = timeit.default_timer() # Init time
            (batchX, batchY) = (batchX.to(self.device), batchY.to(self.device)) # Load data
            if training: self.optimizer.zero_grad() # Zero grad before mini-batch
            loss = self.loss_compute(batchX, batchY) # Forward model

            # Optimizer step and Backpropagation
            if training:
                loss.backward()
                self.optimizer.step()
            
            # Loss and samples size for evaluation
            sum_loss += loss
            samples += batchY.size(0) # sample size
            batch_cnt += 1

            # Logging
            print(log_template.format(batch_cnt, loss/(2*batchY.size(0)), timeit.default_timer()-start, datetime.datetime.now()))
        
        return sum_loss/(2*samples)
    
    def loss_compute(self, input, output):
        input = torch.reshape(input, (-1, 3, 224, 224)) # Reshape to NxCxHxW
        predict = self.model(input.float())
        loss = self.loss_func(predict.float(), output.float(), output.size(0))
        return loss.item()*(2*output.size(0))

    def save_model(self, curr_epoch, save_path):
        torch.save({
            'epoch': curr_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_arr': self.train_arr,
            'val_arr': self.val_arr
        }, save_path + str(curr_epoch) + '.pt')
    
    def load_model(self, load_path):
        ckpt = torch.load(load_path)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.start_epoch = ckpt['epoch']
        self.train_arr = ckpt['train_arr']
        self.val_arr = ckpt['val_arr']