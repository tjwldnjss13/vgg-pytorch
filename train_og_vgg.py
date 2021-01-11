import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import os
from model import make_conv_layers, vgg_cfgs
import torchvision.models as models

batch_size = 16
learning_rate = .009
num_epoch = 30

data_dir = 'C://DeepLearningData/PoKemonData'
transform_resize = transforms.Compose([transforms.RandomResizedCrop(224), transforms.ToTensor()])
train_data = dset.ImageFolder(data_dir, transform=transform_resize, target_transform=None)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = models.VGG(make_conv_layers(vgg_cfgs['D']), 150).to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_loss_arr = []
for i in range(num_epoch):
    epoch_loss = 0
    batches_per_epoch = 0
    for j, [image, label] in enumerate(train_loader):
        if j == len(train_loader) - 1:
            print('[{}/{} epoch] {}/{}'.format(i + 1, num_epoch, j * batch_size, len(train_data)), end='')
        else:
            print('[{}/{} epoch] {}/{}'.format(i + 1, num_epoch, j * batch_size, len(train_data)), end='\r')
        x = image.to(device)
        y_ = label.to(device)

        optimizer.zero_grad()
        output = model(x)
        loss = loss_func(output, y_)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        batches_per_epoch += 1

    train_loss_arr.append(epoch_loss / batches_per_epoch)

    total = 0
    correct = 0
    with torch.no_grad():
        val_epoch_loss = 0
        val_batches_per_epoch = 0
        for image, label in val_loader:
            x = image.to(device)
            y_ = label.to(device)

            output = model(x)
            val_loss = loss_func(output, y_)
            _, output_idx = torch.max(output, 1)

            total += label.size(0)
            correct += (output_idx == y_).sum().float()

            val_epoch_loss += val_loss.item()
            val_batches_per_epoch += 1

        val_loss_arr.append(val_epoch_loss / val_batches_per_epoch)

    print(' <train_loss> {}  <val_loss> {}'.format(train_loss_arr[-1], val_loss_arr[-1]))

plt.plot(loss_arr)
plt.show()

save_path = './saved_model/vgg16_pokemon1.pth'
# torch.save(model, save_path)

torch.save({
    'epoch': num_epoch,
    'learning_rate': learning_rate,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss_arr
}, save_path)
