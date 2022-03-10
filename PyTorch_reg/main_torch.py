import numpy as np
from PyTorch_reg.dataloader import path_loader
from PyTorch_reg.trainprocedure import fit, test_model
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PyTorch_reg.vgg_simple import VGG_simple

def main():
    # Regression model for
    train_paths, train_labels = path_loader('train', 'reg') 
    val_paths, val_labels = path_loader('val', 'reg')
    test_paths, test_labels = path_loader('test', 'reg')

    # Hyperparameters
    batch_size = 128
    epochs = 24
    loss_func = nn.MSELoss()
    save_path = './PyTorch_reg/vgg_simple_checkpoint/vgg_simple_epoch_'    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = VGG_simple().to(device).float()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    
    # fit(train_data=(train_paths, train_labels), 
    #     val_data=(val_paths, val_labels), 
    #     model=model,
    #     optimizer=optimizer, loss_func=loss_func, 
    #     epochs=epochs, batch_size=batch_size,
    #     save_path=save_path, device=device)

    test_model(test_data=(test_paths, test_labels), 
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