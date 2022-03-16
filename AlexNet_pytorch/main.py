import torch
import matplotlib.pyplot as plt
from AlexNet_Model import AlexNet_Reg, AlexNet_Class
from TrainingProcedure_Reg import Training_Procedure
from DataLoader import Dataloader
from datetime import datetime
from lossFunc import L2_dist


def viz_res(root, dest_folder, trainLoss, val_trainAcc, ars_trainAcc, valLoss, val_valAcc, ars_valAcc, savename=None):
    plt.figure(figsize=(15, 10))

    plt.subplot(1, 2, 1)  # row 1, col 2 index 1
    plt.plot(trainLoss, color='#17bccf', label='Train')
    plt.plot(valLoss, color='#ff7f44', label='Val')
    plt.title("Loss Func")
    plt.xlabel('Epochs')
    plt.grid()
    plt.legend()

    plt.subplot(1, 2, 2)  # index 2
    plt.plot(val_trainAcc, color='#17bccf', label='Valence Train')
    plt.plot(ars_trainAcc, color='#20bccf', label='Arousal Train')
    plt.plot(val_valAcc, color='#ff7f30', label='Valence Val')
    plt.plot(ars_valAcc, color='#ff7f44', label='Valence Val')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    if savename != None:
        plt.savefig(
            root + dest_folder + savename + '.jpg', dpi=500)
    plt.show()


def main():

    root = "../"
    result = "AlexNet_torch_20220314"

    batch_size = 256
    shuffle = True
    num_workers = 0
    regression = True

    train_image_dir = "data/train_set/images/"
    train_reg_frame = "train_reg.csv"
    train_class_frame = "train_class.csv"

    val_image_dir = "data/train_set/images/"
    val_reg_frame = "val_reg.csv"
    val_class_frame = "val_class.csv"

    test_image_dir = "data/val_set/images/"
    test_reg_frame = "test_reg.csv"
    test_class_frame = "test_class.csv"

    model = AlexNet_Reg()

    train_classLoader = Dataloader(root, train_image_dir, train_class_frame).load_batch(
        batch_size, shuffle, num_workers)
    train_regLoader = Dataloader(root, train_image_dir, train_reg_frame, regression=regression).load_batch(
        batch_size, shuffle, num_workers)

    val_classLoader = Dataloader(root, val_image_dir, val_class_frame).load_batch(
        batch_size, shuffle, num_workers)
    val_regLoader = Dataloader(root, val_image_dir, val_reg_frame, regression=regression).load_batch(
        batch_size, shuffle, num_workers)

    test_classLoader = Dataloader(root, test_image_dir, test_class_frame).load_batch(
        batch_size, shuffle, num_workers)
    test_regLoader = Dataloader(root, test_image_dir, test_reg_frame, regression=regression).load_batch(
        batch_size, shuffle, num_workers)

    lr = 0.001
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss = L2_dist

    training_class = Training_Procedure(
        model=model,
        train_dataloader=train_regLoader,
        val_dataloader=val_regLoader,
        test_dataloader=test_regLoader,
        batch_size=batch_size,
        epochs=20,
        optimizer=optimizer,
        loss_func=loss,
        shuffle=True, num_workers=0, lr=lr, savename="output/reg_20220314/"+result, regression=regression
    )

    trainLoss, val_trainRMSE, ars_trainRMSE, valLoss, val_valRMSE, ars_valRMSE = training_class.process()
    viz_res(root, "AlexNet_pytorch/output/", trainLoss, val_trainRMSE, ars_trainRMSE, valLoss, val_valRMSE, ars_valRMSE, result)


print("Start training:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
main()
print("Finish training:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# print(torch.cuda.is_available())
