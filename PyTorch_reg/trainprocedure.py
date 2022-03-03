import torch
import numpy as np
from PIL import Image
import timeit
import matplotlib.pyplot as plt

def batch(inputs, outputs, size):
    # loop over the dataset
    for i in range(0, len(inputs), size):
        # yield a tuple of the current batched data and labels
        input_imgs = []
        for item in inputs[i:i+size]:
            # Convert to image with normalization
            input_imgs.append(np.asarray(Image.open(item).resize((128,128),Image.ANTIALIAS)) / 255) 
        yield (np.stack(input_imgs, axis=0), outputs[i:i + size])

def eval(input, output, 
         optimizer, loss_func, 
         model, training):
    input = torch.reshape(input, (-1, 3, 128, 128))
    predict = model(input.float())
    loss = torch.sqrt(loss_func(predict.float(), output.float()))
    # print(loss)
    if training: 
        optimizer.zero_grad()
        loss.requres_grad = True
        loss.backward()
        optimizer.step()
    return loss.item()

def fit(train_data, val_data, 
        model,
        optimizer, loss_func, 
        epochs, batch_size,
        save_path, device):
    # Train and Val data
    x_train, y_train = train_data
    x_val, y_val = val_data

    # Initialize loss
    train_loss = []
    val_loss = []

    for epoch in range(1, epochs+1):
        model.train()
        samples = 0
        train_avg_loss = 0
        val_avg_loss = 0

        # Training 
        for (batchX, batchY) in batch(x_train, y_train, batch_size):
            start = timeit.default_timer()
            (batchX, batchY) = (torch.from_numpy(batchX).to(device), torch.from_numpy(batchY).to(device))
            loss = eval(input=batchX, output=batchY, optimizer=optimizer, loss_func=loss_func, model=model, training=True)
            train_avg_loss += loss
            samples += 1
            print('Mini-batch {}: {} - {}s'.format(samples, loss, timeit.default_timer()-start))
        train_avg_loss = train_avg_loss / samples # Average loss across mini-batch
        train_loss.append(train_avg_loss)

        
        # Evaluation
        samples = 0
        model.eval() # Canceling regularization
        for (batchX, batchY) in batch(x_val, y_val, batch_size):
            (batchX, batchY) = (torch.from_numpy(batchX).to(device), torch.from_numpy(batchY).to(device))
            val_avg_loss += eval(input=batchX, output=batchY, optimizer=optimizer, loss_func=loss_func, model=model, training=False)
            samples += 1
        val_avg_loss = val_avg_loss / samples # Average loss across mini-batch
        val_loss.append(val_avg_loss)
        
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss
                    }, save_path+str(epoch)+'.pt')

        # display model progress on the current training batch
        template = "epoch: {} train loss: {} val loss: {}"
        print(template.format(epoch, train_avg_loss, val_avg_loss))

def test_model(test_data, 
                model,
                optimizer, loss_func, 
                batch_size,
                load_path, device):
    x_test, y_test = test_data
    
    ckpt = torch.load(load_path)
    plt.plot(ckpt['train_loss'])
    plt.plot(ckpt['val_loss'])
    plt.grid()
    plt.show()

    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    test_avg_loss = 0
    samples = 0
    model.eval() # Canceling regularization
    for (batchX, batchY) in batch(x_test, y_test, batch_size):
        (batchX, batchY) = (torch.from_numpy(batchX).to(device), torch.from_numpy(batchY).to(device))
        test_avg_loss += eval(input=batchX, output=batchY, optimizer=optimizer, loss_func=loss_func, model=model, training=False)
        samples += 1
    test_avg_loss = test_avg_loss / samples # Average loss across mini-batch
    print('Test loss: ', test_avg_loss)


    