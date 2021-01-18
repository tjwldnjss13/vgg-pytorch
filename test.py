import numpy as np
import torch
import torchvision.transforms as transforms

from PIL import Image


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    class_dict = {'background': 0, 'aeroplane': 1, 'bicycle': 2, 'bird': 3, 'boat': 4, 'bottle': 5, 'bus': 6, 'car': 7,
                  'cat': 8, 'chair': 9, 'cow': 10, 'diningtable': 11, 'dog': 12, 'horse': 13, 'motorbike': 14, 'person': 15,
                  'pottedplant': 16, 'sheep': 17, 'sofa': 18, 'train': 19, 'tvmonitor': 20, 'ambigious': 255}

    model_pth = 'saved models/15epoch_0.0001lr_4.64132loss_0.48529acc.pth'
    model = torch.load(model_pth).to(device)
    model.eval()

    img_pth = 'sample/boat.jpg'
    img = Image.open(img_pth)

    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    x = transform(img).unsqueeze(0).to(device)

    output = model(x)
    output = torch.argmax(output).item()

    for key, value in class_dict.items():
        if output == value:
            print(key)
            break
