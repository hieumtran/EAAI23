import torch
import numpy as np
from PIL import Image
import timeit
import matplotlib.pyplot as plt
import datetime


class procedure:
    def __init__(self, optimizer, loss_func, model, epoch, batch_size, device):
        self.optimizer = optimizer  # optimizer
        self.loss_func = loss_func # loss function
        self.model = model  # model init
        self.device = device  # device
        self.batch_size = batch_size  # mini-batch size
        self.epoch = epoch  # epoch

    def fit(self, train_data, val_data):
        # Train and Val data
        x_train, y_train = train_data
        x_val, y_val = val_data

        # Initialize loss
        train_loss = []
        val_loss = []

        for epoch in range(1, epochs + 1):
            self.model.train()
            train_samples = 0
            val_samples = 0
            batch_train = 0
            batch_val = 0
            avg_train_loss = 0
            avg_val_loss = 0

    def train_val(self, input, output, arr_loss):
        samples = 0
        batch = 0
        avg_loss = 0

        for


def batch(inputs, outputs, size):
    # loop over the dataset
    for i in range(0, len(inputs), size):
        # yield a tuple of the current batched data and labels
        input_imgs = []
        for item in inputs[i:i + size]:
            # Convert to image with normalization
            input_imgs.append(np.asarray(Image.open(item)) / 255)
        yield np.stack(input_imgs, axis=0), outputs[i:i + size]


def eval(input, output,
         optimizer, loss_func,
         model, training):
    input = torch.reshape(input, (-1, 3, 224, 224))
    predict = model(input.float())
    loss = loss_func(predict.float(), output.float(), output.size(0))
    if training:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item() * (2 * output.size(0))


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

    for epoch in range(1, epochs + 1):
        model.train()
        train_samples = 0
        val_samples = 0
        batch_train = 0
        batch_val = 0
        avg_train_loss = 0
        avg_val_loss = 0

        # Training
        # Output Template
        train_templ = 'Training mini-batch {}: {:.10f} / Time: {:.5f}s / Current date & Time: {:%Y-%m-%d %H:%M:%S}'
        for (batchX, batchY) in batch(x_train, y_train, batch_size):
            start = timeit.default_timer()
            (batchX, batchY) = (torch.from_numpy(batchX).to(device), torch.from_numpy(batchY).to(device))
            loss = eval(input=batchX, output=batchY, optimizer=optimizer, loss_func=loss_func, model=model,
                        training=True)
            avg_train_loss += loss
            train_samples += batchY.size(0)
            batch_train += 1
            print(train_templ.format(batch_train, loss / (batchY.size(0) * 2), timeit.default_timer() - start,
                                     datetime.datetime.now()))
        train_loss.append(loss / (batchY.size(0) * 2))

        # Evaluation
        # Output Template
        val_templ = 'Validation mini-batch {}: {:.10f} / Time: {:.5f}s / Current date & Time: {:%Y-%m-%d %H:%M:%S}'
        model.eval()  # Canceling regularization
        for (batchX, batchY) in batch(x_val, y_val, batch_size):
            start = timeit.default_timer()
            (batchX, batchY) = (torch.from_numpy(batchX).to(device), torch.from_numpy(batchY).to(device))
            loss = eval(input=batchX, output=batchY, optimizer=optimizer, loss_func=loss_func, model=model,
                        training=False)
            avg_val_loss += loss
            val_samples += batchY.size(0)
            batch_val += 1
            print(val_templ.format(batch_val, loss / (batchY.size(0) * 2), timeit.default_timer() - start,
                                   datetime.datetime.now()))
        val_loss.append(loss / (batchY.size(0) * 2))

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
        }, save_path + str(epoch) + '.pt')

        # display model progress on the current training batch
        template = 'Epoch {} / train: {:.10f} / val: {:.10f}  / Time: {:.5f}s / Current date & Time: {:%Y-%m-%d ' \
                   '%H:%M:%S} '
        print(template.format(epoch, avg_train_loss / (train_samples * 2), avg_val_loss / (val_samples * 2),
                              timeit.default_timer() - start, datetime.datetime.now()))


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
    plt.savefig('./resnet.jpg')
    # plt.show()

    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    avg_test_total_loss = 0
    avg_test_val_loss = 0
    avg_test_ars_loss = 0
    test_samples = 0
    model.eval()  # Canceling regularization
    for (batchX, batchY) in batch(x_test, y_test, batch_size):
        (batchX, batchY) = (torch.from_numpy(batchX).to(device), torch.from_numpy(batchY).to(device))
<<<<<<< HEAD
        loss, val_loss, ars_loss = eval(input=batchX, output=batchY, optimizer=optimizer, loss_func=loss_func,
                                        model=model, training=False)
=======
        loss = eval(input=batchX, output=batchY, optimizer=optimizer, loss_func=loss_func, model=model, training=False)
>>>>>>> 7c835e6a5d5d7d2ced54dd98ff9fad00039b8632
        avg_test_total_loss += loss
        avg_test_val_loss += val_loss
        avg_test_ars_loss += ars_loss
        print(batchY.size())
        test_samples += batchY.size(0)
    template = 'Test loss: {} / val_loss: {} / ars_loss: {} '
    print(template.format(np.sqrt(avg_test_total_loss / test_samples), np.sqrt(avg_test_val_loss / test_samples),
                          np.sqrt(avg_test_ars_loss / test_samples)))
