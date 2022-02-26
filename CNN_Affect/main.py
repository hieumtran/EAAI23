from alexnet_var_model import *
from Training_Procedure import *
import matplotlib.pyplot as plt


def viz_res(trainLoss, trainAcc, valLoss, valAcc, savename=None):
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
        plt.savefig('./figure/' + savename + '.jpg', dpi=500)
    plt.show()


def main():
    model = alexnet_var_model()
    train_input_path = "C:/Phanh/BuAnhNet/EAAI23/data/train_set/New Folder"
    train_output = np.load("C:/Phanh/BuAnhNet/EAAI23/data/trainval_class.npy")

    test_input_path = "C:/Phanh/BuAnhNet/EAAI23/data/val_set/New Folder"
    test_output = np.load(
        "C:/Phanh/BuAnhNet/EAAI23/data/test_class_sorted.npy")
    batch_size = 400
    epochs = 24

    training_procedure = Training_Procedure(
        model,
        image_size=128,
        train_input_path=train_input_path,
        train_output=train_output,
        test_input_path=test_input_path,
        test_output=test_output,
        batch_size=batch_size,
        epochs=epochs,
        regression=False
    )

    model.save('AlexNet_Variant_20220227')
    hist = training_procedure.process()

    train_loss = hist.history['loss']
    train_acc = hist.history['accuracy']
    val_loss = hist.history['val_loss']
    val_acc = hist.history['val_accuracy']

    viz_res(train_loss, train_acc, val_loss, val_acc, 'alexnet_var_20220227')


main()
