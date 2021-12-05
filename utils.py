import os
import math
import shutil
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt


def show_img(tensor, label='',  zoom=3, color_map='gray'):

    img_channels, img_width, img_height = tensor.shape

    print('******************', label, '**************************')
    print("Número de canales:", img_channels)
    print("Dimensiones:", img_width, "x", img_height)

    rows = cols = math.ceil(np.sqrt(img_channels))

    fig_grid = plt.figure(figsize=(rows*zoom, cols*zoom))

    for i in range(0, rows*cols):
        fig_grid.add_subplot(rows, cols, i+1)
        if i < img_channels:
            plt.imshow(tensor[i].detach(), cmap=color_map)

    plt.show()


def copyFiles(source, to_dir):
    for name in source:
        shutil.copy(name, to_dir)


def split_dataset(root_dir, dest_dir, test_ratio=0.2, val_ratio=0.1):
    train_dir = os.path.join(dest_dir, 'train')
    test_dir = os.path.join(dest_dir, 'test')
    val_dir = os.path.join(dest_dir, 'val')

    if len(os.listdir(dest_dir)) > 0:
        print("Ya existen datos de train, test y val.")
        return train_dir, test_dir, val_dir

    print("########### Splitting Dataset ###########")
    for cls in os.listdir(root_dir):
        cls_dir = os.path.join(root_dir, cls)

        # Validar que el elemento es un folder sino iterar al siguiente.
        if os.path.isdir(cls_dir) == False:
            continue

        print("Clase: " + cls)

        cls_img_path = [os.path.join(cls_dir, name)
                        for name in os.listdir(cls_dir)]

        # Barajamos las imágenes de la clase
        np.random.shuffle(cls_img_path)

        train_len = int(len(cls_img_path) * (1 - (test_ratio + val_ratio)))
        test_len = int(len(cls_img_path) * test_ratio)

        # Dividimos el total de imágenes en 3 grupos: train, test, val
        train_img_path, test_img_path, val_img_path = np.split(
            cls_img_path, [train_len, (train_len + test_len)])

        print('total:', len(cls_img_path))

        # Copy train images
        print('train:', len(train_img_path))
        train_cls_dir = os.path.join(train_dir, cls)
        os.makedirs(train_cls_dir)
        copyFiles(train_img_path, train_cls_dir)

        # Copy test images
        print('test:', len(test_img_path))
        test_cls_dir = os.path.join(test_dir, cls)
        os.makedirs(test_cls_dir)
        copyFiles(test_img_path, test_cls_dir)

        # Copy validation images
        print('val:',  len(val_img_path))
        val_cls_dir = os.path.join(val_dir, cls)
        os.makedirs(val_cls_dir)
        copyFiles(val_img_path, val_cls_dir)

    print("########### ########### ###########")
    return train_dir, test_dir, val_dir


def size_output_layer(width, kernel, stride, padding):
    # PISO[((2P + N - k ) / S ) +  1]
    return math.floor(((width - kernel + (2*padding)) / stride) + 1)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layers = nn.ModuleList([])

    def addLayer(self, layer):
        self.layers.append(layer)

    def addClassifier(self, classifier):
        self.fc = classifier

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)

        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
