import torch
import timeit
import matplotlib.pyplot as plt
import datetime
from loss_function import rmse, pear, ccc, sagr

class procedure:
    def __init__(self, optimizer, scheduler,
                 loss_func, model, start_epoch, end_epoch,
                 save_path, save_fig, device):
        self.optimizer = optimizer  # optimizer
        self.scheduler = scheduler  # scheduler for optimizer
        self.loss_func = loss_func  # loss function
        self.model = model  # model init
        self.device = device  # device
        self.start_epoch = start_epoch  # Beginning epoch
        self.end_epoch = end_epoch  # Ending epoch
        self.train_arr = []  # Init training loss
        self.val_arr = []  # Init validation loss
        self.save_path = save_path  # Checkpoint directory
        self.save_fig = save_fig  # Figure saving
        self.scheduler = scheduler  # scheduler for optimizer

    def fit(self, train_loader, val_loader):
        output_template = 'Epoch {} |' \
                          'train: {:.8f} | val: {:.8f} |' \
                          'Time: {:.5f}s | Current date & Time: {:%Y-%m-%d %H:%M:%S}'
        for epoch in range(self.start_epoch, self.end_epoch + 1):
            epoch_start = timeit.default_timer()

            # Update loss and optimizer
            train_loss = self.train_val_test(train_loader, state='train')
            self.scheduler.step(train_loss)
            self.train_arr.append(train_loss)

            if val_loader != None:
                val_loss = self.train_val_test(val_loader, state='val')
                self.val_arr.append(val_loss)
            else: val_loss = 0

            self.save_model(epoch)  # Saving model
            print(output_template.format(epoch, train_loss, val_loss,
                                         timeit.default_timer() - epoch_start,
                                         datetime.datetime.now()))

    def train_val_test(self, loader, state):
        assert state in ['train', 'test', 'val']
        if state == 'train':
            self.model.train()
            init_log = 'Train '
        elif state == 'val':
            self.model.eval()
            init_log = 'Val '
        else:
            self.model.eval()
            init_log = 'Test '
            
        log_template = init_log + 'mini-batch {}: {:.8f} | Time: {:.5f}s | Current date & Time: {:%Y-%m-%d %H:%M:%S}'
        predict = []
        truth = []
        
        # Init computation variables
        samples, batch_cnt, sum_loss = (0, 0, 0)
        for (batchX, batchY) in loader:
            start = timeit.default_timer()
            (batchX, batchY) = (batchX.to(self.device).float(), batchY.to(self.device).float())  # Load data2
            if state == 'train': self.optimizer.zero_grad()  # Zero grad before mini-batch
            pred, loss = self.loss_compute(batchX, batchY)  # Forward model

            # Optimizer step and Backpropagation
            if state == 'train':
                loss.backward()
                self.optimizer.step()

            # Loss and samples size for evaluation
            sum_loss += loss.item() * (2 * batchY.size(0))
            samples += batchY.size(0)  # sample size
            batch_cnt += 1
            
            # Concat predict and truth value
            if state == 'test':
                with torch.no_grad():
                    predict.append(pred.to('cpu'))
                    truth.append(batchY.to('cpu'))

            # Logging
            print(log_template.format(batch_cnt, loss.item(), timeit.default_timer() - start, datetime.datetime.now()))
        if state == 'train' or state == 'val': return sum_loss / (2 * samples)
        else: return sum_loss / (2 * samples), torch.concat(predict), torch.concat(truth)

    def loss_compute(self, input, output):
        input = torch.reshape(input, (-1, 3, 224, 224))  # Reshape to NxCxHxW
        predict = self.model(input.float())
        output = output.reshape(-1, 2)
        loss = self.loss_func(predict.float(), output.float())
        return predict, loss

    def test(self, test_loader):
        start = timeit.default_timer()
        output_template = 'Test: {:.8f} | RMSE_Val: {:.8f} | RMSE_Ars: {:.8f} | P_Val: {:.8f} | P_Ars: {:.8f} |' \
                            ' C_Val: {:.8f} | C_Ars: {:.8f} | S_Val: {:.8f} | S_Ars: {:.8f} |' 
        time_template =  'Time: {:.5f}s | Current date & Time: {:%Y-%m-%d %H:%M:%S}'
        output_template += time_template
        test_loss, predict, truth = self.train_val_test(test_loader, state='test')
        
        truth = truth.reshape(-1, 2)
        rmse_val, rmse_ars = rmse(predict, truth)
        pear_val, pear_ars = pear(predict, truth)
        ccc_val, ccc_ars = ccc(predict, truth)
        sagr_val, sagr_ars = sagr(predict, truth)
        
        print(output_template.format(test_loss, rmse_val, rmse_ars, pear_val, pear_ars, ccc_val, ccc_ars, sagr_val, sagr_ars,
                                     timeit.default_timer() - start, datetime.datetime.now()))

    def save_model(self, curr_epoch):
        torch.save({
            'epoch': curr_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_arr': self.train_arr,
            'val_arr': self.val_arr
        }, self.save_path + str(curr_epoch) + '.pt')

    def load_model(self, load_path):
        ckpt = torch.load(load_path)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.train_arr = ckpt['train_arr']
        self.val_arr = ckpt['val_arr']

    def visualize(self, save_fig):
        plt.plot(self.train_arr, label='Train loss')
        plt.plot(self.val_arr, label='Test loss')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(save_fig, dpi=500)
        # plt.show()
