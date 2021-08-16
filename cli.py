import os
import sys
import cv2
import torch
import imgaug
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imgaug import augmenters
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader as DL

import utils as u
from api import fit, predict
from Model import build_model
from Dataset import DS

#########################################################################################################

def get_images_and_labels_from_csv(path=None):
    assert(path is not None)

    data = pd.read_csv(path, engine="python")
    return data.iloc[:, 1:].copy().values.astype("uint8"), data.iloc[:, 0].copy().values

#########################################################################################################

def build_train_and_valid_loaders(batch_size=None, augment=False):
    images, labels = get_images_and_labels_from_csv(os.path.join(u.DATA_PATH, "Train.csv"))

    if augment:
        augment = get_dataset_augment(seed=u.SEED)
        images = augment(images=images)
    
    tr_images, va_images, tr_labels, va_labels = train_test_split(images, labels, 
                                                                  test_size=0.2, 
                                                                  shuffle=True, 
                                                                  random_state=u.SEED, 
                                                                  stratify=labels)
    tr_data_setup = DS(X=tr_images, y=tr_labels.reshape(-1, 1), transform=u.TRANSFORM, mode="train")
    va_data_setup = DS(X=va_images, y=va_labels.reshape(-1, 1), transform=u.TRANSFORM, mode="valid")
    tr_data = DL(tr_data_setup, batch_size=batch_size, shuffle=True, generator=torch.manual_seed(u.SEED))
    va_data = DL(va_data_setup, batch_size=batch_size, shuffle=False)
    dataloaders = {"train" : tr_data, "valid" : va_data}
    return dataloaders


def build_test_loader():
    pass

#########################################################################################################
def save_graphs(L, A) -> None:
    TL, VL, TA, VA = [], [], [], []

    for i in range(len(L)):
        TL.append(L[i]["train"])
        VL.append(L[i]["valid"])
        TA.append(A[i]["train"])
        VA.append(A[i]["valid"])
    
    x_Axis = np.arange(1, len(L)+1, 1)
    plt.figure("Graphs")
    plt.subplot(1, 2, 1)
    plt.plot(x_Axis, TL, "r", label="Train")
    plt.plot(x_Axis, VL, "b", label="Valid")
    plt.grid()
    plt.legend()
    plt.title("Loss Graph")
    plt.subplot(1, 2, 2)
    plt.plot(x_Axis, TA, "r", label="Train")
    plt.plot(x_Axis, VA, "b", label="Valid")
    plt.grid()
    plt.legend()
    plt.title("Accuracy Graph")
    plt.savefig("./Graphs.jpg")
    plt.close("Graphs")

#########################################################################################################

def get_dataset_augment(seed=None):
    imgaug.seed(seed)
    augment = augmenters.Sequential([
        augmenters.VerticalFlip(p=0.25),
        augmenters.HorizontalFlip(p=0.25),
        augmenters.Affine(scale=(0.8, 1.2), translate_percent=(-0.2, 0.2), rotate=(-45, 45), seed=seed),
    ])

    return augment

#########################################################################################################

def app():
    args_1 = "--hl"
    args_2 = "--fs"
    args_3 = "--epochs"
    args_4 = "--lr"
    args_5 = "--wd"
    args_6 = "--bs"
    args_7 = "--early"

    epochs = 10
    HL = None
    filter_sizes = [4, 4, 4]
    lr = 1e-3
    wd = 0
    batch_size = 64
    early_stopping = 5

    if args_1 in sys.argv:
        if sys.argv[sys.argv.index(args_1) + 1] == "1":
            HL = [int(sys.argv[sys.argv.index(args_1) + 2])]
        elif sys.argv[sys.argv.index(args_1) + 1] == "2":
            HL = [int(sys.argv[sys.argv.index(args_1) + 2]), 
                  int(sys.argv[sys.argv.index(args_1) + 3])]
    if args_2 in sys.argv:
        filter_sizes = [int(sys.argv[sys.argv.index(args_2) + 2]), 
                        int(sys.argv[sys.argv.index(args_2) + 3]),
                        int(sys.argv[sys.argv.index(args_2) + 4])]
    if args_3 in sys.argv:
        epochs = int(sys.argv[sys.argv.index(args_3) + 1])
    if args_4 in sys.argv:
        lr = float(sys.argv[sys.argv.index(args_4) + 1])
    if args_5 in sys.argv:
        wd = float(sys.argv[sys.argv.index(args_5) + 1])
    if args_6 in sys.argv:
        batch_size = int(sys.argv[sys.argv.index(args_6) + 1])
    if args_7 in sys.argv:
        early_stopping = int(sys.argv[sys.argv.index(args_7) + 1])
    
    dataloaders = build_train_and_valid_loaders(batch_size=batch_size)
    model = build_model(filter_sizes=filter_sizes, HL=HL)
    optimizer = model.getOptimizer(lr=lr, wd=wd)

    L, A, _, _ = fit(model=model, optimizer=optimizer, scheduler=None, epochs=epochs,
                     dataloaders=dataloaders, early_stopping_patience=early_stopping, verbose=True)
    save_graphs(L, A)

#########################################################################################################