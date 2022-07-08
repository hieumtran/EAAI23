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
    transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5686, 0.4505, 0.3990],std=[0.2332, 0.2064, 0.1956]), data_augment])
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5686, 0.4505, 0.3990],std=[0.2332, 0.2064, 0.1956])])

    train = Dataloader(root=root, image_dir=train_image_dir, label_frame=train_reg_frame, subset=subset, transform=transform_train,
                       mode='reg').load_batch(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test = Dataloader(root=root, image_dir=test_image_dir, label_frame=test_reg_frame, subset=subset, transform=transform_test,
                      mode='reg').load_batch(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return train, test


def main():
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)
    torch.backends.cudnn.benchmark = True

    # Data parameters
    batch_size = 8
    num_workers = 4
    subset = None
    # subset = 100
    root_dir = './'
    shuffle = False
    train_reg_frame = "train_subset_20000_reg.csv"
    test_reg_frame = "test_reg.csv"
    train_loader, test_loader = load_data(root=root_dir, train_reg_frame=train_reg_frame, test_reg_frame=test_reg_frame,
                                            batch_size=batch_size, subset=subset, 
                                            num_workers=num_workers, shuffle=shuffle)

    # Model parameters
    start_epoch = 0
    end_epoch = 50
    # save_path = './PyTorch_reg/design/InvNet_weight/InvNet101_small_'
    # save_path = './PyTorch_reg/design/InvNet_weight/InvNet17_small_'
    save_path = './PyTorch_reg/design/InvNet_weight/InvNet50_large_AdamW_'
    save_fig = './PyTorch_reg/figure/InvNet_aug'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # model = MobileInvNet(input_channel, final_channel, block_setting, 3).to(device)
    # model = InvNet(3, [32, 64, 128, 256], [3, 4, 23, 2], 0.5, 7).to(device) # InvNet101
    # model = InvNet(3, [32, 64, 128, 256], [1, 1, 1, 1], 0.5, 7).to(device)
    # model = InvNet(3, [64, 128, 256, 512], [3, 4, 6, 2], 0.5, 7).to(device)
    model = InvNet(3, [64, 128, 256, 512], [3, 4, 6, 2], 0.5, 7).to(device) # InvNet50
    breakpoint()
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total number of parameters: ', pytorch_total_params)
    print('Dataset name: ' + train_reg_frame)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, eps=1e-8, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    # procedure init
    proceed = procedure(optimizer=optimizer, scheduler=scheduler, model=model,
                        start_epoch=start_epoch, end_epoch=end_epoch, device=device,
                        save_path=save_path, save_fig=save_fig)
    # proceed.load_model('./PyTorch_reg/design/InvNet_weight/InvNet17_small_30.pt')
    # proceed.test(test_loader)
    # proceed.fit(train_loader, test_loader)
    for i in range(0, 51):
        proceed.load_model(save_path + str(i) + '.pt')
        proceed.val(test_loader, i)
    proceed.visualize('./PyTorch_reg/figure/InvNet50_large_AdamW_loss.jpg')
