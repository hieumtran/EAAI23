import torch
import matplotlib.pyplot as plt
from AlexNet_Model import AlexNet_Reg, AlexNet_Class
from Training_Procedure import Training_Procedure
from DataLoader import Dataloader


def viz_res(root, dest_folder, trainLoss, trainAcc, valLoss, valAcc, savename=None):
    plt.figure(figsize=(15, 7))

    plt.subplot(1, 2, 1)  # row 1, col 2 index 1
    plt.plot(trainLoss, color='#17bccf', label='Train')
    plt.plot(valLoss, color='#ff7f44', label='Val')
    plt.title("Loss Func")
    plt.xlabel('Epochs')
    plt.grid()
    plt.legend()

    plt.subplot(1, 2, 2)  # index 2
    plt.plot(trainAcc, color='#17bccf', label='Train')
    plt.plot(valAcc, color='#ff7f44', label='Val')
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

    root = "C:/Phanh/BuAnhNet/EAAI23/"
    result = "AlexNet_torch_20220307"

    batch_size = 32
    shuffle = True
    num_workers = 0

    train_image_dir = "data/train_set/images/"
    train_reg_frame = "train_reg.csv"
    train_class_frame = "train_class.csv"

    val_image_dir = "data/train_set/images/"
    val_reg_frame = "val_reg.csv"
    val_class_frame = "val_class.csv"

    test_image_dir = "data/val_set/images/"
    test_reg_frame = "test_reg.csv"
    test_class_frame = "test_class.csv"

    model = AlexNet_Class()
    train_classLoader = Dataloader(root, train_image_dir, train_class_frame).load_batch(
        batch_size, shuffle, num_workers)
    train_regLoader = Dataloader(root, train_image_dir, train_reg_frame, regression=True).load_batch(
        batch_size, shuffle, num_workers)

    val_classLoader = Dataloader(root, val_image_dir, val_class_frame).load_batch(
        batch_size, shuffle, num_workers)
    val_regLoader = Dataloader(root, val_image_dir, val_reg_frame, regression=True).load_batch(
        batch_size, shuffle, num_workers)

    test_classLoader = Dataloader(root, test_image_dir, test_class_frame).load_batch(
        batch_size, shuffle, num_workers)
    test_regLoader = Dataloader(root, test_image_dir, test_reg_frame, regression=True).load_batch(
        batch_size, shuffle, num_workers)

    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = torch.nn.NLLLoss()

    training_reg = Training_Procedure(
        model=model,
        train_dataloader=train_classLoader,
        val_dataloader=val_classLoader,
        test_dataloader=test_classLoader,
        batch_size=batch_size,
        epochs=20,
        optimizer=optimizer,
        loss_func=loss,
        shuffle=True, num_workers=0, lr=lr, savename=result
    )

    trainLoss, trainAcc, valLoss, valAcc = training_reg.process()
    viz_res(root, result+"/", trainLoss, trainAcc, valLoss, valAcc, result)


main()
# print(torch.cuda.is_available())
