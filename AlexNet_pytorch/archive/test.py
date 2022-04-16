import torch
import matplotlib.pyplot as plt
from AlexNet_Model import AlexNet_Reg, AlexNet_Class
from DataLoader import Dataloader, Dataset


def main():
    root = "../"
    train_image_dir = "data/train_set/images/"
    train_reg_frame = "train_reg.csv"
    train_class_frame = "train_class.csv"

    train_classLoader = Dataloader(
        root, train_image_dir, train_reg_frame, regression=True)
    batch_iter = train_classLoader.load_batch(1024, True, 0)

    for x, y in batch_iter:
        print(type(x), type(y))
        print(x.shape, y.shape)
        # y = y.reshape(-1, 2)
        print(y.shape)
        break


main()
