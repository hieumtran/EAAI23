from alexnet_var_model import *
from Training_Procedure import *
import matplotlib.pyplot as plt
import os


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
        plt.savefig(
            'C:/Phanh/BuAnhNet/EAAI23/AlexNet_Variant_20220227/' + savename + '.jpg', dpi=500)
    plt.show()


def main():
    model = alexnet_var_model()
    train_input_path = "C:/Phanh/BuAnhNet/EAAI23/data/train_set/New Folder"
    train_output = np.load("C:/Phanh/BuAnhNet/EAAI23/data/trainval_class.npy")

    test_input_path = "C:/Phanh/BuAnhNet/EAAI23/data/val_set/New Folder"
    test_output = np.load(
        "C:/Phanh/BuAnhNet/EAAI23/data/test_class_sorted.npy")
    batch_size = 256
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
    hist, test_accuracy, params_cnt = training_procedure.process()

    train_loss = hist['loss']
    train_acc = hist['accuracy']
    val_loss = hist['val_loss']
    val_acc = hist['val_accuracy']

    viz_res(train_loss, train_acc, val_loss, val_acc, 'alexnet_var_20220227')


# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# with tf.device('/gpu:0'):
#     main()

# print(tf.__version__)

# print(tf.config.list_physical_devices('GPU'))

trainLoss = [
    3.3178, 2.7960, 2.6149, 2.5106, 2.4304, 2.3739,
    2.3218, 2.2776, 2.2419, 2.2035, 2.1751, 2.1457,
    2.1197, 2.0970, 2.0739, 2.0526, 2.0269, 2.0060,
    1.9825, 1.9630, 1.9487, 1.9274, 1.9079, 1.9076
]

trainAcc = [
    0.4990, 0.5067, 0.5210, 0.5495, 0.5778, 0.5869,
    0.5923, 0.5973, 0.5986, 0.5976, 0.6015, 0.6054,
    0.6067, 0.6067, 0.6100, 0.6118, 0.6077, 0.6050,
    0.6057, 0.6025, 0.6008, 0.5941, 0.5511, 0.5822
]


valLoss = [
    3.0105, 2.7138, 2.5906, 2.4882, 2.4154, 2.4438,
    2.3566, 2.3207, 2.2851, 2.2745, 2.2735, 2.3047,
    2.2972, 2.2413, 2.2370, 2.2601, 2.2403, 2.2806,
    2.2437, 2.2603, 2.2557, 2.2256, 2.3428, 2.2763
]


valAcc = [
    0.5150, 0.8725, 0.7053, 0.4961, 0.8958, 0.8661,
    0.6943, 0.6299, 0.6157, 0.6550, 0.6408, 0.6343,
    0.6584, 0.6024, 0.6144, 0.7470, 0.6433, 0.7304,
    0.6183, 0.7297, 0.6314, 0.5856, 0.7162, 0.6534
]

viz_res(trainLoss=trainLoss, trainAcc=trainAcc, valLoss=valLoss,
        valAcc=valAcc, savename='alexnet_var_20220227')
