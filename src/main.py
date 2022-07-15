import numpy as np
from dataloader import Dataloader
from procedure import procedure
import torch
from design.InvNeXt.model import InvNet
from torchvision import transforms
import pandas as pd
from sklearn.utils import class_weight

def L2_loss(pred, truth):
    return torch.sum((pred - truth) ** 2) / (2 * truth.size(0))

def load_data(config):
    # Transform argument
    data_augment = transforms.RandomApply([transforms.GaussianBlur(kernel_size=(7, 13), sigma=(2, 5)),
                                            transforms.RandomHorizontalFlip(p=0.5), 
                                            transforms.ColorJitter(brightness=(0.5,1.5), contrast=(1), saturation=(0.5,1.5), hue=(-0.1,0.1)),
                                            transforms.RandomErasing()
                                            ], p=0.5)
    if config.augmentation:
        transform_train = transforms.Compose([transforms.ToTensor(), 
                                            transforms.Normalize(mean=[0.5686, 0.4505, 0.3990],std=[0.2332, 0.2064, 0.1956]),
                                            data_augment])
    else:
        transform_train = transforms.Compose([transforms.ToTensor(), 
                                            transforms.Normalize(mean=[0.5686, 0.4505, 0.3990],std=[0.2332, 0.2064, 0.1956])])

    transform_test = transforms.Compose([transforms.ToTensor(), 
                                        transforms.Normalize(mean=[0.5686, 0.4505, 0.3990],std=[0.2332, 0.2064, 0.1956])])

    train = Dataloader(
        root=config.root, 
        image_dir=config.train_image_dir, 
        label_frame=config.train_input, 
        subset=config.subset, 
        transform=transform_train,
        mode=config.mode,
        config=config
     ).load_batch(batch_size=config.batch_size, shuffle=config.shuffle, num_workers=config.num_workers)

    test = Dataloader(
        root=config.root, 
        image_dir=config.test_image_dir, 
        label_frame=config.test_input, 
        subset=config.subset, 
        transform=transform_test,
        mode=config.mode,
        config=config
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
    if config.mode == 'reg':
        loss = L2_loss

    elif config.mode == 'class':
        num_sample = pd.read_csv(f'{config.root}data/{config.train_input}', names=['images', 'class'])['class']
        class_weights=class_weight.compute_class_weight('balanced', classes=np.unique(num_sample), y=num_sample.to_numpy())
        class_weights=torch.tensor(class_weights,dtype=torch.double)
        loss = torch.nn.CrossEntropyLoss(weight=class_weights.to(config.device))

    elif config.mode == 'class_reg':
        num_sample = pd.read_csv(f'{config.root}data/{config.train_input}', names=['images', 'class', 'val', 'ars'])['class']
        class_weights=class_weight.compute_class_weight('balanced', classes=np.unique(num_sample), y=num_sample.to_numpy())
        class_weights=torch.tensor(class_weights,dtype=torch.float)
        entropy_loss = torch.nn.CrossEntropyLoss(weight=class_weights.to(config.device))
        # entropy_loss = torch.nn.CrossEntropyLoss()
        mse_loss = torch.nn.MSELoss()
        # loss = (entropy_loss, L2_loss)
        loss = (entropy_loss, mse_loss)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('Total number of parameters: ', pytorch_total_params)
    
    if config.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
    if config.optimizer == 'Adam':
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, eps=1e-8, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    # procedure init
    proceed = procedure(optimizer=optimizer, scheduler=scheduler, loss=loss, model=model,
                        start_epoch=config.start_epoch, end_epoch=config.end_epoch, 
                        device=config.device, mode=config.mode, val_ars=config.val_ars,
                        save_path=config.save_path, save_name=config.save_name,
                        accumulative_iteration=config.accummulative_iteration)
    
    # Training further
    if config.transfer_file != None: proceed.load_model(f'{config.save_path}{config.transfer_file}')

    # Training
    if config.task == 'train': proceed.fit(train_loader, test_loader)

    # Testing
    if config.task == 'test':
        for i in range(config.start_epoch, config.end_epoch + 1):
            proceed.load_model(f'{config.save_path}{config.save_name}{i}.pt')
            proceed.test(test_loader=test_loader, epoch=i)
