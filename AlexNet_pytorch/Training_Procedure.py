import torch
import timeit
from DataLoader import Dataloader


class Training_Procedure:

    def __init__(
        self, model,
        train_dataloader, val_dataloader, test_dataloader,
        batch_size, epochs, optimizer, loss_func,
        shuffle=False, num_workers=0, lr=0.001, savename=None
    ):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.model = model.to(self.device)

        self.train_batchIter = train_dataloader
        self.val_batchIter = val_dataloader
        self.test_batchIter = test_dataloader

        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.lr = lr
        self.savename = savename

    # training
    def train(self, x_train, y_train):
        # setting all the gradient to zero
        self.optimizer.zero_grad()

        # training
        train_output = self.model(x_train)

        # calculating loss
        train_loss = self.loss_func(train_output, y_train)
        train_loss.backward()

        self.optimizer.step()

        # adding parameters' gradients to their values, multiplied by the learning rate
        for p in self.model.parameters():
            p.data.add_(p.grad.data, alpha=-self.lr)

        return train_output, train_loss.item()

    # validation and testing by batch (loop through batches in this function)

    # evaluating after training in each epoch
    def evaluate(self, x_val, y_val):
        with torch.no_grad():
            # feed validation set through the model
            val_output = self.model(x_val)

            # calculating validation loss
            val_loss = self.loss_func(val_output, y_val)

            return val_output, val_loss.item()

    # testing
    def test(self, x_test, y_test):
        return self.evaluate(x_test, y_test)

    def accuracy_cnt(self, y_pred, y_true):
        acc_cnt = (y_pred.max(1)[1] == y_true).sum().item()
        return acc_cnt

    def process(self):

        trainLoss_glob = []
        trainAcc_glob = []
        valLoss_glob = []
        valAcc_glob = []

        for epoch in range(self.epochs):

            start_epoch = timeit.default_timer()

            # initialize tracker variables and set our model to trainable
            trainLoss = 0
            trainAcc = 0
            valLoss = 0
            valAcc = 0

            y_train_len = 0
            y_val_len = 0

            # training by batches
            for (batchX, batchY) in self.train_batchIter:
                batchX = batchX.to(self.device)
                batchY = batchY.to(self.device)
                train_y_predicted, loss = self.train(batchX, batchY)
                trainLoss += loss
                trainAcc += self.accuracy_cnt(train_y_predicted, batchY)
                y_train_len += batchY.shape[0]

            # saving training results
            avg_trainLoss = trainLoss / y_train_len
            avg_trainAcc = trainAcc / y_train_len

            trainLoss_glob.append(avg_trainLoss)
            trainAcc_glob.append(avg_trainAcc)

            # feeding the validation batches into the model
            for (batchX, batchY) in self.val_batchIter:
                batchX = batchX.to(self.device)
                batchY = batchY.to(self.device)
                val_y_predicted, loss = self.evaluate(batchX, batchY)
                valLoss += loss
                valAcc += self.accuracy_cnt(val_y_predicted, batchY)
                y_val_len += batchY.shape[0]

            # saving validation results
            avg_valLoss = valLoss / y_val_len
            avg_valAcc = valAcc / y_val_len

            valLoss_glob.append(avg_valLoss)
            valAcc_glob.append(avg_valAcc)

            # display model progress on the current training batch
            template = "epoch: {} time: {} train loss: {} train accuracy: {} val loss: {} val acc: {}"
            print(template.format(epoch + 1, timeit.default_timer()-start_epoch, avg_trainLoss,
                  avg_trainAcc, avg_valLoss, avg_valAcc))

        testLoss = 0
        testAcc = 0
        y_test_len = 0

        # Test dataset
        for (batchX, batchY) in self.test_batchIter:
            batchX = batchX.to(self.device)
            batchY = batchY.to(self.device)
            test_y_predicted, loss = self.evaluate(batchX, batchY)
            testLoss += loss
            testAcc += self.accuracy_cnt(test_y_predicted, batchY)
            y_test_len += batchY.shape[0]

        # Saving Evaluation Results
        avg_testLoss = testLoss / y_test_len
        avg_testAcc = testAcc / y_test_len

        # display test model
        print('Test loss: {} Test acc: {}'.format(avg_testLoss, avg_testAcc))

        torch.save(self.model.state_dict(), self.savename+".pth")

        return trainLoss_glob, trainAcc_glob, valLoss_glob, valAcc_glob
