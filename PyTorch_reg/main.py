import numpy as np
from PyTorch_reg.dataloader import Dataloader
from PyTorch_reg.procedure import procedure
import torch
import torch.nn as nn
# from design.vgg_simple import VGG_simple # VGG-16
# from design.sp_trans import sp_trans # Spatial Transformer 
from design.resnet import ResNet
from loss_function import L2_dist


def load_data(root, batch_size, num_workers, shuffle):
    # Data loading
    train_image_dir = "data/train_set/images/"
    train_reg_frame = "train_reg.csv"

    val_image_dir = "data/train_set/images/"
    val_reg_frame = "val_reg.csv"

    test_image_dir = "data/val_set/images/"
    test_reg_frame = "test_reg.csv"

    train_loader = Dataloader(root=root, image_dir=train_image_dir, label_frame=train_reg_frame, regression=True).load_batch(
                                batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = Dataloader(root=root, image_dir=val_image_dir, label_frame=val_reg_frame, regression=True).load_batch(
                                batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = Dataloader(root=root, image_dir=test_image_dir, label_frame=test_reg_frame, regression=True).load_batch(
                                batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader

def main():
    # Data parameters
    batch_size = 64
    num_workers = 2
    root_dir = './'
    shuffle = False
    train_loader, val_loader, test_loader = load_data(root=root_dir, batch_size=batch_size, \
                                                        num_workers=num_workers, shuffle=shuffle)
    
    # Model parameters
    start_epoch = 1
    end_epoch = 24
    loss_func = L2_dist
    save_path = './PyTorch_reg/design/resnet/resnet_'
    save_fig = './PyTroch_reg/figure/ResNet_loss'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Init model & Optimizer
    res_net = ResNet(256, [64, 256], 1).to(device).float()
    optimizer = torch.optim.SGD(res_net.parameters(), lr=1e-3)

    # procedure init
    proced = procedure(optimizer=optimizer, loss_func=loss_func, model=res_net, \
                        start_epoch=start_epoch, end_epoch=end_epoch, device=device, \
                        save_path=save_path, save_fig=save_fig)
    proced.fit(train_loader, val_loader)

if __name__ == '__main__':
    main()
