import torch
from torch.utils.data import ConcatDataset
from dataset.voc_dataset import VOCDataset

root = 'C://DeepLearningData/VOC2012/'
img_size = (224, 224)

dset1 = VOCDataset(root, img_size)
dset2 = VOCDataset(root, img_size)
dset3 = VOCDataset(root, img_size)
dset4 = ConcatDataset([dset1, dset2, dset3])

print(len(dset1))
print(len(dset2))
print(len(dset3))
print(len(dset4))