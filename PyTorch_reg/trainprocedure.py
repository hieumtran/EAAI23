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
    val_loss = loss_func(predict[:, 0].float(), output[:, 0].float())
    ars_loss = loss_func(predict[:, 1].float(), output[:, 1].float())
    loss = loss_func(predict.float(), output.float())
    # print(loss)
    if training: 
        optimizer.zero_grad()
        loss.requres_grad = True
        loss.backward()
        optimizer.step()
    return loss.item(), val_loss.item(), ars_loss.item()

def fit(train_data, val_data, 
        model,
        optimizer, loss_func, 
        epochs, batch_size,
        save_path, device):
    # Train and Val data
    x_train, y_train = train_data
    x_val, y_val = val_data

    # Initialize loss
    train_loss_total = []
    train_loss_val = []
    train_loss_ars = []
    val_loss_total = []
    val_loss_val = []
    val_loss_ars = []

    for epoch in range(1, epochs+1):
        model.train()

        train_samples = 0
        val_samples = 0
        batch_train = 0
        batch_val = 0

        avg_train_total_loss = 0
        avg_train_val_loss = 0
        avg_train_ars_loss = 0
        avg_val_total_loss = 0
        avg_val_val_loss = 0
        avg_val_ars_loss = 0
        
        train_templ = 'Training {}: total: {} / val: {} / ars: {} / Time: {}s'
        val_templ = 'Validation {}: total: {} / val: {} / ars: {} / Time: {}s'
        # Training
        for (batchX, batchY) in batch(x_train, y_train, batch_size):
            start = timeit.default_timer()
            (batchX, batchY) = (torch.from_numpy(batchX).to(device), torch.from_numpy(batchY).to(device))
            loss, val_loss, ars_loss = eval(input=batchX, output=batchY, optimizer=optimizer, loss_func=loss_func, model=model, training=True)
            avg_train_total_loss += loss
            avg_train_val_loss += val_loss
            avg_train_ars_loss += ars_loss
            train_samples += batchY.size(0)
            batch_train += 1
            print(train_templ.format(batch_train, np.sqrt(loss/batchY.size(0)), np.sqrt(val_loss/batchY.size(0)), np.sqrt(ars_loss/batchY.size(0)), timeit.default_timer()-start))
        train_loss_total.append(np.sqrt(avg_train_total_loss / train_samples))
        train_loss_val.append(np.sqrt(avg_train_val_loss / train_samples))
        train_loss_ars.append(np.sqrt(avg_train_ars_loss / train_samples))

        
        # Evaluation
        model.eval() # Canceling regularization
        for (batchX, batchY) in batch(x_val, y_val, batch_size):
            start = timeit.default_timer()
            (batchX, batchY) = (torch.from_numpy(batchX).to(device), torch.from_numpy(batchY).to(device))
            loss, val_loss, ars_loss = eval(input=batchX, output=batchY, optimizer=optimizer, loss_func=loss_func, model=model, training=False)
            avg_val_total_loss += loss
            avg_val_val_loss += val_loss
            avg_val_ars_loss += ars_loss
            val_samples += batchY.size(0)
            batch_val += 1
            print(val_templ.format(batch_val, np.sqrt(loss/batchY.size(0)), np.sqrt(val_loss/batchY.size(0)), np.sqrt(ars_loss/batchY.size(0)), timeit.default_timer()-start))
        val_loss_total.append(np.sqrt(avg_val_total_loss / val_samples))
        val_loss_val.append(np.sqrt(avg_val_val_loss / val_samples))
        val_loss_ars.append(np.sqrt(avg_val_ars_loss / val_samples))
        
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss_total': train_loss_total,
                    'train_loss_val': train_loss_val,
                    'train_loss_ars': train_loss_ars,
                    'val_loss': val_loss,
                    'val_loss_val': val_loss_val,
                    'val_loss_ars': val_loss_ars
                    }, save_path+str(epoch)+'.pt')

        # display model progress on the current training batch
        template = 'Epoch {}: train_total: {} / train_al: {} / train_ars: {} / val_total: {} / val_al: {} / val_ars: {}'
        print(template.format(epoch, np.sqrt(avg_train_total_loss / train_samples), np.sqrt(avg_train_val_loss / train_samples), np.sqrt(avg_train_ars_loss / train_samples), np.sqrt(avg_val_total_loss / val_samples), np.sqrt(avg_val_val_loss / val_samples), np.sqrt(avg_val_ars_loss / val_samples)))

def test_model(test_data, aspect, 
                model,
                optimizer, loss_func, 
                batch_size,
                load_path, device):
    x_test, y_test = test_data
    
    ckpt = torch.load(load_path)
    plt.plot(ckpt[aspect[0]])
    plt.plot(ckpt[aspect[1]])
    plt.grid()
    plt.savefig('./sp_trans.jpg')
    # plt.show()

    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    avg_test_total_loss = 0
    avg_test_val_loss = 0
    avg_test_ars_loss = 0
    test_samples = 0
    model.eval() # Canceling regularization
    for (batchX, batchY) in batch(x_test, y_test, batch_size):
        (batchX, batchY) = (torch.from_numpy(batchX).to(device), torch.from_numpy(batchY).to(device))
        loss, val_loss, ars_loss = eval(input=batchX, output=batchY, optimizer=optimizer, loss_func=loss_func, model=model, training=False)
        avg_test_total_loss += loss
        avg_test_val_loss += val_loss
        avg_test_ars_loss += ars_loss
        print(batchY.size())
        test_samples += batchY.size(0)
    template = 'Test loss: {} / val_loss: {} / ars_loss: {} '
    print(template.format(np.sqrt(avg_test_total_loss/ test_samples), np.sqrt(avg_test_val_loss/ test_samples), np.sqrt(avg_test_ars_loss/ test_samples))) 


    