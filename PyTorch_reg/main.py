import numpy as np
from dataloader import Dataloader
from procedure import procedure
import torch
# from design.MobileInvNet.model import MobileInvNet
from design.InvNeXt.model import InvNet
from torchvision import transforms


def load_data(root, train_reg_frame, test_reg_frame, batch_size, num_workers, subset, shuffle):
    # Data loading
    train_image_dir = "data/train_set/images/"
    # train_reg_frame = "train_subset_resampling_10000_reg.csv"
    # train_reg_frame = "train_subset_reg.csv"
    

    test_image_dir = "data/val_set/images/"
    data_augment = transforms.RandomApply([transforms.GaussianBlur(kernel_size=(7, 13), sigma=(2, 5)),
                                            transforms.RandomHorizontalFlip(p=0.5), 
                                            transforms.ColorJitter(brightness=(0.5,1.5), contrast=(1), saturation=(0.5,1.5), hue=(-0.1,0.1)),
                                            transforms.RandomErasing()
                                            ], p=0.5)
    transform = transforms.Compose([transforms.ToTensor(), data_augment,
                                    transforms.Normalize(mean=[0.5686, 0.4505, 0.3990],std=[0.2332, 0.2064, 0.1956])])

    train = Dataloader(root=root, image_dir=train_image_dir, label_frame=train_reg_frame, subset=subset, transform=transform,
                       mode='reg').load_batch(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test = Dataloader(root=root, image_dir=test_image_dir, label_frame=test_reg_frame, subset=subset, transform=transform,
                      mode='reg').load_batch(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return train, test


def main():
    # Data parameters
    batch_size = 256
    num_workers = 0
    subset = None
    # subset = 1000
    root_dir = './'
    shuffle = False
    train_reg_frame = "train_downsample_50000_reg.csv"
    test_reg_frame = "test_reg.csv"
    train_loader, test_loader = load_data(root=root_dir, train_reg_frame=train_reg_frame, test_reg_frame=test_reg_frame,
                                            batch_size=batch_size, subset=subset, 
                                            num_workers=num_workers, shuffle=shuffle)

    # Model parameters
    start_epoch = 0
    end_epoch = 100
    save_path = './PyTorch_reg/design/InvNet_weight/InvNet50_'
    save_fig = './PyTorch_reg/figure/InvNet_aug'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # model = MobileInvNet(input_channel, final_channel, block_setting, 3).to(device)
    model = InvNet(3, [16, 32, 64, 256], [3, 4, 6, 2], 0.5, 1).to(device)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total number of parameters: ', pytorch_total_params)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    # procedure init
    proceed = procedure(optimizer=optimizer, scheduler=scheduler, model=model,
                        start_epoch=start_epoch, end_epoch=end_epoch, device=device,
                        save_path=save_path, save_fig=save_fig)
    # proceed.load_model('./PyTorch_reg/design/InvNet/InvNet101_aug_68.pt')
    # proceed.test(test_loader)
    proceed.fit(train_loader, test_loader)
    # for i in range(0, 51):
    #     proceed.load_model('./PyTorch_reg/design/InvNet/InvNet18_' + str(i) + '.pt')
    #     proceed.test(test_loader)
    # proceed.visualize('./PyTorch_reg/figure/InvNet101_aug_loss.jpg')
