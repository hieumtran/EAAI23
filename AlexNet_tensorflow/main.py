import numpy as np
import os
import matplotlib.pyplot as plt
from Training_Procedure import *
from alexnet_var_model import *

path = "C:/Users/DePauw/AppData/Local/RenderCard/EAAI23/"


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
            path+'AlexNet_Variant_20220306/' + savename + '.jpg', dpi=500)
    plt.show()


def main():
    model = alexnet_var_model()
    # model = tf.keras.models.load_model(path + "AlexNet_Variant_20220226")
    train_input_path = path + "data/train_set/New Folder"
    train_output = np.load(path + "data/archive/trainval_class.npy")

    test_input_path = path + "data/val_set/New Folder"
    test_output = np.load(
        path + "data/archive/test_class_sorted.npy")
    batch_size = 256
    epochs = 100

    training_procedure = Training_Procedure(
        model,
        image_size=224,
        train_input_path=train_input_path,
        train_output=train_output,
        test_input_path=test_input_path,
        test_output=test_output,
        batch_size=batch_size,
        epochs=epochs,
        regression=False
    )

    hist, testLoss, testAccuracy, params_cnt = training_procedure.process()
    model.save('AlexNet_Variant_20220306')

    train_loss = hist['loss']
    train_acc = hist['accuracy']
    val_loss = hist['val_loss']
    val_acc = hist['val_accuracy']

    print("Test loss: {:5.2f}, Test accuracy: {:5.2f}".format(
        testLoss, testAccuracy))
    print("total parameter count: ", params_cnt)

    viz_res(train_loss, train_acc, val_loss, val_acc, 'alexnet_var_202200306')


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
with tf.device('/gpu:0'):
    main()

# print(tf.__version__)

# print(len(tf.config.list_physical_devices('GPU')))
# print(tf.test.is_gpu_available(cuda_only=True))
