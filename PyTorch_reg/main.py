import numpy as np
from dataloader import Dataloader
from procedure import procedure
import torch
# from design.MobileInvNet.model import MobileInvNet
from design.InvNeXt.model import InvNet
from torchvision import transforms


def load_data(config):
    # Data loading
    data_augment = transforms.RandomApply([transforms.GaussianBlur(kernel_size=(7, 13), sigma=(2, 5)),
                                            transforms.RandomHorizontalFlip(p=0.5), 
                                            transforms.ColorJitter(brightness=(0.5,1.5), contrast=(1), saturation=(0.5,1.5), hue=(-0.1,0.1)),
                                            transforms.RandomErasing()
                                            ], p=0.5)
    # transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5686, 0.4505, 0.3990],std=[0.2332, 0.2064, 0.1956]), data_augment])
    transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5686, 0.4505, 0.3990],std=[0.2332, 0.2064, 0.1956])])
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5686, 0.4505, 0.3990],std=[0.2332, 0.2064, 0.1956])])

    train = Dataloader(
        root=config.root, 
        image_dir=config.train_image_dir, 
        label_frame=config.train_input, 
        subset=config.subset, 
        transform=transform_train,
        mode=config.mode
     ).load_batch(batch_size=config.batch_size, shuffle=config.shuffle, num_workers=config.num_workers)
    test = Dataloader(
        root=config.root, 
        image_dir=config.test_image_dir, 
        label_frame=config.test_input, 
        subset=config.subset, 
        transform=transform_test,
        mode=config.mode
    ).load_batch(batch_size=config.batch_size, shuffle=config.shuffle, num_workers=config.num_workers)

    return train, test


def main(config):
    # Speed up training process
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)
    torch.backends.cudnn.benchmark = True

    # Load data
    train_loader, test_loader = load_data(config)

    # Model parameters
    start_epoch = 0
    end_epoch = 50

    model = InvNet(
        in_channel=config.in_channel, 
        dims=config.dims, 
        num_per_layers=config.num_per_layers, 
        dropout_rate=config.dropout_rate,
        inv_kernel=config.inv_kernel
    ).to(config.device) 

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total number of parameters: ', pytorch_total_params)
    print('Dataset name: ' + config.train_input)

    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, eps=1e-8, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    # procedure init
    proceed = procedure(optimizer=optimizer, scheduler=scheduler, model=model,
                        start_epoch=start_epoch, end_epoch=end_epoch, device=config.device,
                        save_path=config.save_path, save_name=config.save_name,
                        accumulative_iteration=config.accummulative_iteration)
    # proceed.fit(train_loader, test_loader)
    for i in range(0, 51):
        proceed.load_model(f'{config.save_path}{config.save_name}{i}.pt')
        proceed.val(test_loader, i)
    # proceed.visualize('./PyTorch_reg/figure/InvNet50_large_AdamW_loss.jpg')
