import numpy as np
from dataloader import Dataloader
from procedure import procedure
import torch
import torch.nn as nn
# from design.vgg_simple import VGG_simple # VGG-16
# from design.sp_trans import sp_trans # Spatial Transformer 
from design.resnet import ResNet
from loss_function import L2_dist


def load_data(root, batch_size, subset, num_workers, shuffle):
    # Data loading
    train_image_dir = "data/train_set/images/"
    train_reg_frame = "train_reg.csv"

    val_image_dir = "data/train_set/images/"
    val_reg_frame = "val_reg.csv"

    test_image_dir = "data/val_set/images/"
    test_reg_frame = "test_reg.csv"

    train = Dataloader(root=root, image_dir=train_image_dir, subset=subset, label_frame=train_reg_frame,
                       regression=True).load_batch(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val = Dataloader(root=root, image_dir=val_image_dir, subset=subset, label_frame=val_reg_frame,
                     regression=True).load_batch(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test = Dataloader(root=root, image_dir=test_image_dir, subset=subset, label_frame=test_reg_frame,
                      regression=True).load_batch(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return train, val, test


def main():
    # Data parameters
    batch_size = 32
    num_workers = 0
    subset = None
    root_dir = './'
    shuffle = False
    train_loader, val_loader, test_loader = load_data(root=root_dir, batch_size=batch_size,
                                                      subset=subset, num_workers=num_workers, shuffle=shuffle)

    # Model parameters
    start_epoch = 1
    end_epoch = 24
    loss_func = L2_dist
    save_path = './PyTorch_reg/design/resnet/resnet_'
    save_fig = './PyTorch_reg/figure/ResNet_loss'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Init model & Optimizer
    res_net = ResNet(res_learning=[3, 4, 6, 3]).to(device).float()
    optimizer = torch.optim.SGD(res_net.parameters(), lr=1e-1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    # procedure init
    proceed = procedure(optimizer=optimizer, scheduler=scheduler,
                        loss_func=loss_func, model=res_net,
                        start_epoch=start_epoch, end_epoch=end_epoch, device=device,
                        save_path=save_path, save_fig=save_fig)
    proceed.fit(train_loader, val_loader)


if __name__ == '__main__':
    main()
