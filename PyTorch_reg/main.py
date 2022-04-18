import numpy as np
from dataloader import Dataloader
from procedure import procedure
import torch
import torch.nn as nn
# from design.vgg_simple import VGG_simple # VGG-16
# from design.sp_trans import sp_trans # Spatial Transformer 
# from design.resnet import ResNet
from design.simplenet import simpleNet
from loss_function import L2_dist


def load_data(root, batch_size, num_workers, subset, shuffle, validation):
    # Data loading
    train_image_dir = "data/train_set/images/"
    train_reg_frame = "train_subset_reg.csv"

    val_image_dir = "data/train_set/images/"
    val_reg_frame = "val_reg.csv"

    test_image_dir = "data/val_set/images/"
    test_reg_frame = "test_reg.csv"

    train = Dataloader(root=root, image_dir=train_image_dir, label_frame=train_reg_frame, subset=subset,
                       regression=True).load_batch(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val = Dataloader(root=root, image_dir=val_image_dir, label_frame=val_reg_frame, subset=subset,
                     regression=True).load_batch(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test = Dataloader(root=root, image_dir=test_image_dir, label_frame=test_reg_frame, subset=subset,
                      regression=True).load_batch(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    if validation == False: val = None
    return train, val, test


def main():
    # Data parameters
    batch_size = 8
    num_workers = 0
    subset = None
    # subset = 1000
    root_dir = './'
    shuffle = False
    train_loader, val_loader, test_loader = load_data(root=root_dir, batch_size=batch_size, subset=subset, 
                                                      num_workers=num_workers, shuffle=shuffle, validation=False)

    # Model parameters
    start_epoch = 0
    end_epoch = 100
    loss_func = L2_dist
    save_path = './PyTorch_reg/design/simplenet/simplenet_last13_'
    save_fig = './PyTorch_reg/figure/simplenet_last13_loss'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Init model & Optimizer
    # res_net = ResNet(res_learning=[3, 4, 23, 3]).to(device).float()
    # res_net = ResNet(res_learning=[3, 4, 6, 3]).to(device).float()
    # optimizer = torch.optim.SGD(res_net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)

    simple_net = simpleNet(3, 2).to(device)
    optimizer = torch.optim.SGD(simple_net.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    # procedure init
    proceed = procedure(optimizer=optimizer, scheduler=scheduler,
                        loss_func=loss_func, model=simple_net,
                        start_epoch=start_epoch, end_epoch=end_epoch, device=device,
                        save_path=save_path, save_fig=save_fig)
    # proceed.load_model('./PyTorch_reg/design/simplenet/simplenet_200.pt')
    # proceed.test(test_loader)
    proceed.fit(train_loader, val_loader)
    # for i in range(1, 250):
        # proceed.load_model('./PyTorch_reg/design/simplenet/simplenet_' + str(i) + '.pt')
        # proceed.test(test_loader)
    # proceed.visualize('./PyTorch_reg/figure/simplenet_loss.jpg')
        
if __name__ == '__main__':
    main()
