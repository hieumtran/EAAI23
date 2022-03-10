import numpy as np
from dataloader import DataLoader, path_loader # Uncertain on DataLoader - Anh
from trainprocedure import fit, test_model # Wrong test_model function
import torch
import torch.nn as nn
# from design.vgg_simple import VGG_simple # VGG-16
# from design.sp_trans import sp_trans # Spatial Transformer 
from design.resnet import ResNet
from loss_function import L2_dist

def main():
    # Regression model for
    train_paths, train_labels = path_loader('train', 'reg') 
    val_paths, val_labels = path_loader('val', 'reg')
    test_paths, test_labels = path_loader('test', 'reg')

    # Hyperparameters
    batch_size = 64
    epochs = 24
    loss_func = L2_dist
    save_path = './PyTorch_reg/design/resnet/resnet_'    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ResNet(256, [64, 256], 1).to(device).float()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    
    # fit(train_data=(train_paths, train_labels), 
    #     val_data=(val_paths, val_labels), 
    #     model=model,
    #     optimizer=optimizer, loss_func=loss_func, 
    #     epochs=epochs, batch_size=batch_size,
    #     save_path=save_path, device=device)

    test_model(test_data=(test_paths, test_labels),  aspect = ['train_loss', 'val_loss'],
                model=model,
                optimizer=optimizer, loss_func=loss_func, 
                batch_size=batch_size,
                load_path=save_path + '24.pt', device=device)
    


    # for (batchX, batchY) in batch(train_paths[:5], train_labels[:5], 1):
    #     plt.imshow(batchX.squeeze())
    #     plt.title('Valence: {} / Arousal: {}'.format(batchY.squeeze()[0], batchY.squeeze()[1]))
    #     plt.show()

if __name__ == '__main__':
    main()