import torch
import timeit
import numpy as np
import datetime
from eval_metrics import *
from torch.cuda.amp import GradScaler, autocast

class procedure:
    def __init__(self, optimizer, scheduler, loss, model, 
                 start_epoch, end_epoch, mode,
                 save_path, save_name,
                 accumulative_iteration, device):
        self.optimizer = optimizer  # optimizer
        self.scheduler = scheduler  # scheduler for optimizer
        self.loss = loss # Loss function
        self.model = model  # model init
        self.device = device  # device

        self.mode = mode# Task: reg or class

        self.start_epoch = start_epoch  # Beginning epoch
        self.end_epoch = end_epoch  # Ending epoch
        
        self.train_arr = []  # Init training loss

        self.save_path = save_path # Save directory
        self.save_name = save_name  # Checkpoint directory

        self.accumulative_iteration = accumulative_iteration # Accumulative iteration
        self.scheduler = scheduler  # scheduler for optimizer
        self.scaler = GradScaler() # Grad scaler for faster training time

    def fit(self, train_loader, test_loader):
        output_template = 'Epoch {} | train: {:.2f} | Time: {:.5f}s | Current date & Time: {:%Y-%m-%d %H:%M:%S}'
        for epoch in range(self.start_epoch, self.end_epoch + 1):
            epoch_start = timeit.default_timer()

            # Update loss and optimizer
            train_loss = self.train_test(train_loader, state='train')
            self.scheduler.step()
            self.train_arr.append(train_loss)

            self.save_model(epoch)  # Saving model
            print(output_template.format(epoch, train_loss, timeit.default_timer() - epoch_start, datetime.datetime.now()))

            self.test(test_loader, epoch)

    def train_test(self, loader, state):
        assert state in ['train', 'test']
        if state == 'train':
            self.model.train()
            init_log = 'Train '
            log_template = init_log + 'mini-batch {}: {:.2f} | Time: {:.5f}s | Current date & Time: {:%Y-%m-%d %H:%M:%S}'
        else:
            self.model.eval()
            
        predict = []
        truth = []

        # Init computation variables
        samples, sum_loss = (0, 0)
        self.model.zero_grad()
        for batch_idx, (batchX, batchY) in enumerate(loader):
            start = timeit.default_timer()
            (batchX, batchY) = (batchX.to(self.device).float(), batchY.to(self.device).float())  # Load data2

            pred, loss = self.loss_compute(batchX, batchY)  # Forward model
            
            # Normalize loss to account for batch accumulation
            loss = loss / self.accumulative_iteration 

            # Optimizer step and Backpropagation
            if state == 'train':
                self.scaler.scale(loss).backward()
                if ((batch_idx + 1) % self.accumulative_iteration == 0) or (batch_idx + 1 == len(loader)):
                    self.scaler.step(self.optimizer)
                    self.model.zero_grad(set_to_none=True)
                    self.scaler.update()

                # Logging
                print(log_template.format(batch_idx, loss.item() * self.accumulative_iteration, timeit.default_timer() - start, datetime.datetime.now()))

            # Loss and samples size for evaluation
            sum_loss += loss.item() * self.accumulative_iteration * (2 * batchX.size(0))
            samples += batchX.size(0)  # sample size
            
            # Concat predict and truth value
            if state == 'test':
                with torch.no_grad():
                    # Convert to cpu for easier computation
                    if self.mode != 'class_reg': predict.append(pred.to('cpu'))
                    else: predict.append(torch.concat([pred[0], pred[1]], axis=1).to('cpu'))
                    truth.append(batchY.to('cpu'))

        if state == 'train': 
            return sum_loss / (2 * samples)
        else: 
            return sum_loss / (2 * samples), torch.concat(predict).cpu().detach().numpy(), torch.concat(truth).cpu().detach().numpy()

    def loss_compute(self, input, output):
        with autocast():
            input = torch.reshape(input, (-1, 3, 224, 224))  # Reshape to NxCxHxW
            pred = self.model(input.float(), self.mode)

            if self.mode == 'reg':
                output = output.reshape(-1, 2) # Reshape to Nx2
                loss = self.loss(pred.float(), output.float())

            elif self.mode == 'class':
                loss = self.loss(pred.double(), output.type(torch.LongTensor).to(self.device))
            
            elif self.mode == 'class_reg':
                class_loss = self.loss[0](pred[0], output[:, 0].type(torch.LongTensor).to(self.device)) / output.shape[0]
                reg_loss = self.loss[1](pred[1].float(), output[:, 1:].float())
                loss = class_loss + reg_loss

        return pred, loss

    def test(self, test_loader, epoch):
        start = timeit.default_timer()
 
        time_template =  'Time: {:.5f}s | Current date & Time: {:%Y-%m-%d %H:%M:%S}'
        
        test_loss, predict, truth = self.train_test(test_loader, state='test')
 
        if self.mode == 'reg':            
            output_template = 'Val {}: {:.2f} | RMSE_Val: {:.2f} | RMSE_Ars: {:.2f} | ' \
                              'P_Val: {:.2f} | P_Ars: {:.2f} | ' \
                              'C_Val: {:.2f} | C_Ars: {:.2f} | S_Val: {:.2f} | S_Ars: {:.2f} |'
            output_template += time_template

            truth = truth.reshape(-1, 2)
            rmse_val, rmse_ars = rmse(predict, truth)
            pear_val, pear_ars = pear(predict, truth)
            ccc_val, ccc_ars = ccc(predict, truth)
            sagr_val, sagr_ars = sagr(predict, truth)
            
            print(output_template.format(epoch, test_loss, rmse_val, rmse_ars, 
                                        pear_val, pear_ars, ccc_val, ccc_ars, sagr_val, sagr_ars,
                                        timeit.default_timer() - start, datetime.datetime.now()))
        elif self.mode == 'class': 
            output_template = 'Val {}: {:.2f} | Accuracy: {:.2f} | F1_score: {:.2f} | '
            output_template += time_template

            predict = np.argmax(predict, 1)
            acc = accuracy(predict, truth)
            f1_score = f1_score_func(predict, truth)

            print(output_template.format(epoch, test_loss, acc, f1_score, 
                                        timeit.default_timer() - start, datetime.datetime.now()))
        
        elif self.mode == 'class_reg':
            output_template = 'Val {}: {:.2f} | RMSE_Val: {:.2f} | RMSE_Ars: {:.2f} | ' \
                                'P_Val: {:.2f} | P_Ars: {:.2f} | ' \
                                'C_Val: {:.2f} | C_Ars: {:.2f} | S_Val: {:.2f} | S_Ars: {:.2f} | ' \
                                'Accuracy: {:.2f} | F1_score: {:.2f} | '
            output_template += time_template

            rmse_val, rmse_ars = rmse(predict[:, -2:], truth[:, -2:])
            pear_val, pear_ars = pear(predict[:, -2:], truth[:, -2:])
            ccc_val, ccc_ars = ccc(predict[:, -2:], truth[:, -2:])
            sagr_val, sagr_ars = sagr(predict[:, -2:], truth[:, -2:])

            class_pred = np.argmax(predict[:, :-2], 1)
            acc = accuracy(class_pred, truth[:, 0])
            f1_score = f1_score_func(class_pred, truth[:, 0])

            print(output_template.format(epoch, test_loss, rmse_val, rmse_ars, 
                                         pear_val, pear_ars, ccc_val, ccc_ars, 
                                         sagr_val, sagr_ars, acc, f1_score,
                                         timeit.default_timer() - start, datetime.datetime.now()))

    def save_model(self, curr_epoch):
        torch.save({
            'epoch': curr_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_arr': self.train_arr,
        }, f'{self.save_path}{self.save_name}{curr_epoch}.pt')

    def load_model(self, load_path):
        ckpt = torch.load(load_path)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.train_arr = ckpt['train_arr']

    # def visualize(self, save_fig):
    #     plt.plot(self.train_arr, label='Train loss')
    #     plt.plot(self.val_arr, label='Test loss')
    #     plt.legend()
    #     plt.grid()
    #     plt.tight_layout()
    #     plt.savefig(save_fig, dpi=500)
