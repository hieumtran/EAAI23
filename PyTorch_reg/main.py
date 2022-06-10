import numpy as np
from dataloader import Dataloader
from procedure import procedure
import torch
import torch.nn as nn
from design.simplenet import simpleNet

from design.InvNeXt.model import InvNet


def load_data(root, batch_size, num_workers, subset, shuffle, validation):
    # Data loading
    train_image_dir = "data/train_set/images/"
    # train_reg_frame = "train_subset_resampling_10000_reg.csv"
    # train_reg_frame = "train_subset_reg.csv"
    train_reg_frame = "train_CJ_HF_20000.csv"

    test_image_dir = "data/val_set/images/"
    test_reg_frame = "test.csv"

    train = Dataloader(root=root, image_dir=train_image_dir, label_frame=train_reg_frame, subset=subset,
                       mode='both').load_batch(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test = Dataloader(root=root, image_dir=test_image_dir, label_frame=test_reg_frame, subset=subset,
                      mode='both').load_batch(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    if validation == False: val = None
    return train, val, test


def main():
    # Data parameters
    batch_size = 32
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
    save_path = './PyTorch_reg/design/InvNet/InvNet50_aug_'
    save_fig = './PyTorch_reg/figure/InvNet_aug'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = InvNet(3, [64, 128, 256, 512], [2, 2, 3, 2], 0.5).to('cuda')
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('Total number of parameters: ', pytorch_total_params)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    # procedure init
    proceed = procedure(optimizer=optimizer, scheduler=scheduler, model=model,
                        start_epoch=start_epoch, end_epoch=end_epoch, device=device,
                        save_path=save_path, save_fig=save_fig)
    # proceed.load_model('./PyTorch_reg/design/InvNet/InvNet101_aug_68.pt')
    # proceed.test(test_loader)
    proceed.fit(train_loader, test_loader)
    # for i in range(1, 69):
        # proceed.load_model('./PyTorch_reg/design/InvNet/InvNet101_aug_' + str(i) + '.pt')
        # proceed.test(test_loader)
    # # proceed.load_model('./PyTorch_reg/design/MyDesign/MyDesign_84.pt')
    # proceed.visualize('./PyTorch_reg/figure/InvNet101_aug_loss.jpg')
        
if __name__ == '__main__':
    main()
