import torch
from torch.utils.data import ConcatDataset
from dataset.voc_dataset import VOCDataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_pth = 'pretrained models/vgg_voc_0.0001lr_1.90828loss_0.74948acc.pth'
model = torch.load(model_pth).to(device)
model.classifier = torch.nn.Sequential(*[model.classifier[i] for i in range(7)])
model.classifier[6] = torch.nn.Linear(4096, 20).to(device)

print(model)