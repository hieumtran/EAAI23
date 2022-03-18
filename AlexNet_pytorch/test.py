import torch
import matplotlib.pyplot as plt
from AlexNet_Model import AlexNet_Reg, AlexNet_Class
from Training_Procedure import Training_Procedure
from DataLoader import Dataloader, Dataset


def main():
    root = "C:/Phanh/BuAnhNet/EAAI23/"
    train_image_dir = "data/train_set/images/"
    train_reg_frame = "train_reg.csv"
    train_class_frame = "train_class.csv"

    train_classLoader = Dataloader(
        root, train_image_dir, train_class_frame, regression=False)
    batch_iter = train_classLoader.load_batch(1024, True, 0)

    for x, y in batch_iter:
        print(type(x), type(y))
        break


main()
