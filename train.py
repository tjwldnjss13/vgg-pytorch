import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchsummary import summary

import os
import time
import argparse
from model import VGG, make_conv_layers, vgg_cfgs
from dataset.voc_dataset import VOCDataset, collate_fn
from loss import custom_cross_entropy_loss
from utils import make_batch, time_calculator


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device: {}'.format(torch.cuda.get_device_name()))

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', required=False, type=int, default=32)
    parser.add_argument('--epoch', required=False, type=int, default=200)
    parser.add_argument('--lr', required=False, type=float, default=.0001)

    args = parser.parse_args()

    batch_size = args.batch_size
    learning_rate = args.lr
    num_epoch = args.epoch

    root = 'C://DeepLearningData/VOC2012/'
    original_transforms = transforms.Compose([transforms.RandomCrop((224, 224), pad_if_needed=True),
                                              transforms.ToTensor(),
                                              transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    rotate_transforms = transforms.Compose([transforms.RandomCrop((224, 224), pad_if_needed=True),
                                            transforms.RandomRotation((-30, 30)),
                                            transforms.ToTensor(),
                                            transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    horizontal_flip_transforms = transforms.Compose([transforms.RandomCrop((224, 224), pad_if_needed=True),
                                                     transforms.RandomHorizontalFlip(1),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    vertical_flip_transforms = transforms.Compose([transforms.RandomCrop((224, 224), pad_if_needed=True),
                                                   transforms.RandomVerticalFlip(1),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([.485, .456, .406], [.229, .224, .225])])

    original_dset = VOCDataset(root, img_size=(224, 224), transforms=original_transforms, is_categorical=True)
    rotate_dset = VOCDataset(root, img_size=(224, 224), transforms=rotate_transforms, is_categorical=True)
    horizontal_flip_dset = VOCDataset(root, img_size=(224, 224), transforms=horizontal_flip_transforms, is_categorical=True)
    vertical_flip_dset = VOCDataset(root, img_size=(224, 224), transforms=vertical_flip_transforms, is_categorical=True)

    n_class = original_dset.n_class

    n_data = len(original_dset)
    n_train_data = int(n_data * .7)
    n_val_data = n_data - n_train_data
    tarin_val_ratio = [n_train_data, n_val_data]

    original_dset, dset_val = random_split(original_dset, tarin_val_ratio)
    rotate_dset, _ = random_split(rotate_dset, tarin_val_ratio)
    horizontal_flip_dset, _ = random_split(horizontal_flip_dset, tarin_val_ratio)
    vertical_flip_dset, _ = random_split(vertical_flip_dset, tarin_val_ratio)

    dset_train = ConcatDataset([original_dset, rotate_dset, horizontal_flip_dset, vertical_flip_dset])

    train_loader = DataLoader(dset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(dset_val, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    print('Building model...')
    model = VGG(vgg_cfgs['D'], n_class).to(device)
    # model_pth = 'saved models/base model/vgg_5e-05lr_3.67598loss_0.48738acc.pth'
    # model = torch.load(model_pth).to(device)
    loss_func = custom_cross_entropy_loss
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=.001)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=.001)
    # summary(model, (3, 224, 224))

    print('Training...')
    train_loss_arr = []
    train_acc_arr = []
    val_loss_arr = []
    val_acc_arr = []

    n_train_data = len(dset_train)

    print('Train: {} Validation: {}'.format(len(dset_train), len(dset_val)))

    t_start = time.time()
    for e in range(num_epoch):
        n_images = 0
        n_batch = 0
        train_loss = 0
        train_acc = 0

        model.train()
        t_train_start = time.time()
        for i, (images, labels) in enumerate(train_loader):
            mini_batch = len(images)
            n_images += mini_batch
            n_batch += 1
            print('[{}/{}] {}/{} '.format(e + 1, num_epoch, min(n_images, n_train_data), n_train_data), end='')

            x = make_batch(images).to(device)
            y = make_batch(labels, 'class').to(device)

            output = model(x)

            optimizer.zero_grad()
            loss = loss_func(output, y)
            loss.backward()
            optimizer.step()

            acc = torch.true_divide((output.argmax(dim=1) == y.argmax(dim=1)).sum(), mini_batch)

            train_loss += loss.item()
            train_acc += acc.item()

            t_train_end = time.time()
            H, M, S = time_calculator(t_train_end - t_start)

            print('<loss> {:<20} <acc> {:<20} <loss_avg> {:<20} <acc_avg> {:<20} <time> {:02d}:{:02d}:{:02d}'.format(loss.item(), acc.item(), train_loss / n_batch, train_acc / n_batch, int(H), int(M), int(S)))

        H, M, S = time_calculator(t_train_end - t_train_start)

        train_loss_arr.append(train_loss / n_batch)
        train_acc_arr.append(train_acc / n_batch)

        print('        <train_loss> {:<20} <train_acc> {:<20} <time> {:02d}:{:02d}:{:02d}'.format(train_loss_arr[-1], train_acc_arr[-1], int(H), int(M), int(S)), end='')

        with torch.no_grad():
            val_loss = 0
            val_acc = 0
            n_batch = 0

            model.eval()
            for i, (images, labels) in enumerate(val_loader):
                mini_batch = len(images)
                n_batch += 1

                x = make_batch(images).to(device)
                y = make_batch(labels, 'class').to(device)

                output = model(x)

                loss = loss_func(output, y)
                acc = torch.true_divide((output.argmax(dim=1) == y.argmax(dim=1)).sum(), mini_batch)

                val_loss += loss.item()
                val_acc += acc.item()

            val_loss_arr.append(val_loss / n_batch)
            val_acc_arr.append(val_acc / n_batch)

        print(' <val_loss> {:<20}  <val_acc> {:<20}'.format(val_loss_arr[-1], val_acc_arr[-1]))

        if (e + 1) % 5 == 0:
            PATH = 'saved models/{}epoch_{}lr_{:.5f}loss_{:.5f}acc.pth'.format(e + 1, learning_rate, val_loss_arr[-1], val_acc_arr[-1])
            torch.save(model, PATH)

    x_axis = [i for i in range(num_epoch)]

    plt.figure(0)
    plt.title('Loss')
    plt.plot(x_axis, train_loss_arr, 'r-', label='Train loss')
    plt.plot(x_axis, val_loss_arr, 'b:', label='Validation loss')
    plt.legend()

    plt.figure(1)
    plt.title('Accuracy')
    plt.plot(x_axis, train_acc_arr, 'r-', label='Train Accuracy')
    plt.plot(x_axis, val_acc_arr, 'b:', label='Validation Accuracy')
    plt.legend()

    plt.show()

