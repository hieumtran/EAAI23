import numpy as np
from dataloader import Dataloader
from procedure import procedure
import torch
from design.InvNeXt.model import InvNet
from torchvision import transforms
import pandas as pd

def L2_loss(pred, truth):
    return torch.sum((pred - truth) ** 2) / (2 * truth.size(0))

def load_data(config):
    # Transform argument
    data_augment = transforms.RandomApply([transforms.GaussianBlur(kernel_size=(7, 13), sigma=(2, 5)),
                                            transforms.RandomHorizontalFlip(p=0.5), 
                                            transforms.ColorJitter(brightness=(0.5,1.5), contrast=(1), saturation=(0.5,1.5), hue=(-0.1,0.1)),
                                            transforms.RandomErasing()
                                            ], p=0.5)
    transform_train = transforms.Compose([transforms.ToTensor(), 
                                          transforms.Normalize(mean=[0.5686, 0.4505, 0.3990],std=[0.2332, 0.2064, 0.1956]), 
                                          data_augment])
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

    model = InvNet(
        in_channel=config.in_channel, 
        dims=config.dims, 
        num_per_layers=config.num_per_layers, 
        dropout_rate=config.dropout_rate,
        inv_kernel=config.inv_kernel
    ).to(config.device) 

    print('Dataset name: ' + config.train_input)

    # Trasnfer learning for classification
    if config.mode == 'class':
        # Transfer learning for classification
        ckpt = torch.load(f'{config.save_path}{config.transfer_file}')
        model.load_state_dict(ckpt['model_state_dict'])
        # freeze the layers
        for param in model.parameters(): param.requires_grad = False
        # Modify the last layer
        model.linear = torch.nn.Linear(512, 8).to(config.device)

        # Weighted loss
        num_sample = pd.read_csv(f'{config.root}data/{config.train_input}', names=['images', 'class'])['class'].value_counts()
        min_sample = min(num_sample)
        weighted_loss = []
        for i in range(len(num_sample)): weighted_loss.append(num_sample[i] / min_sample)
        loss = torch.nn.CrossEntropyLoss(weight=torch.tensor(weighted_loss).to(config.device).double())

    if config.mode == 'reg':
        loss = L2_loss

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('Total number of parameters: ', pytorch_total_params)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, eps=1e-8, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    # procedure init
    proceed = procedure(optimizer=optimizer, scheduler=scheduler, loss=loss, model=model,
                        start_epoch=config.start_epoch, end_epoch=config.end_epoch, 
                        device=config.device, mode=config.mode,
                        save_path=config.save_path, save_name=config.save_name,
                        accumulative_iteration=config.accummulative_iteration)
    # Training
    if config.task == 'train': proceed.fit(train_loader)

    # Testing
    if config.task == 'test':
        for i in range(config.start_epoch, config.end_epoch + 1):
            proceed.load_model(f'{config.save_path}{config.save_name}{i}.pt')
            proceed.test(test_loader=test_loader, epoch=i)
