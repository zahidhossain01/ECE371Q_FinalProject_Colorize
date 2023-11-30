import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from torchvision.transforms.functional import to_pil_image
import torchvision.models as models
from skimage.color import rgb2lab, lab2rgb, rgb2gray

import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# CUDA stuff
# https://stackoverflow.com/questions/53695105/why-we-need-image-tocuda-when-we-have-model-tocuda
# https://pytorch.org/docs/stable/notes/cuda.html#cuda-semantics
# either tensor.cuda() or tensor.to(device=cuda) (where cuda=torch.device('cuda'))
BATCH_SIZE = 50
LEARNING_RATE = 1e-2

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = os.listdir(root_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.images[idx])
        img = PIL.Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        img = np.asarray(img)
        img_og = np.copy(img)

        LAB = rgb2lab(img_og)
        LAB = (LAB + 128) / 255
        AB = LAB[:,:,1:3]
        img_gray = rgb2gray(img_og)

        img_gray_tensor = torch.from_numpy(img_gray).unsqueeze(0).float()
        AB_tensor = torch.from_numpy(AB.transpose((2,0,1))).float()

        return (img_gray_tensor, AB_tensor)


class AverageMeter(object):
    '''A handy class from the PyTorch ImageNet tutorial''' 
    def __init__(self):
        self.reset()
    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def to_rgb(grayscale_input, ab_input, save_path=None, save_name=None):
    '''Show/save rgb image from grayscale and ab channels
        Input save_path in the form {'grayscale': '/path/', 'colorized': '/path/'}'''
    plt.clf() # clear matplotlib 
    color_image = torch.cat((grayscale_input, ab_input), 0).numpy() # combine channels
    color_image = color_image.transpose((1, 2, 0))  # rescale for matplotlib
    color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
    color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128   
    color_image = lab2rgb(color_image.astype(np.float64))
    grayscale_input = grayscale_input.squeeze().numpy()
    if save_path is not None and save_name is not None: 
        plt.imsave(arr=grayscale_input, fname='{}{}'.format(save_path['grayscale'], save_name), cmap='gray')
        plt.imsave(arr=color_image, fname='{}{}'.format(save_path['colorized'], save_name))


transform = transforms.Compose([
    transforms.Resize((224,224)),
    # transforms.ToTensor(),
])
train_set = ImageDataset(root_dir="datasets/training/rgb", transform=transform)
val_set = ImageDataset(root_dir="datasets/validation/rgb", transform=transform)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)


examples = enumerate(train_loader)
# batch_idx, batch = next(examples)
# print(f"Batch #{batch_idx} | {len(batch)} elements inside object of type {type(batch)}")
# print(f"batch[0]: {type(batch[0])}, batch[1]: {type(batch[1])}")
batch_idx, (b1, b2) = next(examples)
print(f"Batch #{batch_idx} Types   | b1:{type(b1)}, b2:{type(b2)}")
print(f"Batch #{batch_idx} Lengths | len(b1): {len(b1)}, len(b2): {len(b2)}")

img_gray_tensor, img_ab_tensor = b1[0], b2[0]
print(f"Tensor Shapes | gray:{img_gray_tensor.shape}, ab:{img_ab_tensor.shape}")
# read_img = to_pil_image(img_gray_tensor)
# print(read_img)
# plt.imshow(read_img, cmap='gray', vmin=0, vmax=255)
# plt.show()





resnet = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
class Net(nn.Module):
    def __init__(self, input_size=128):
        super(Net, self).__init__()
        MIDLEVEL_FEATURE_SIZE = 128

        ## First half: ResNet
        resnet = models.resnet18(num_classes=365) 
        # Change first conv layer to accept single-channel (grayscale) input
        resnet.conv1.weight = nn.Parameter(resnet.conv1.weight.sum(dim=1).unsqueeze(1)) 
        # Extract midlevel features from ResNet-gray
        self.midlevel_resnet = nn.Sequential(*list(resnet.children())[0:6])

        ## Second half: Upsampling
        self.upsample = nn.Sequential(     
            nn.Conv2d(MIDLEVEL_FEATURE_SIZE, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2)
        )

    def forward(self, x):
        y = self.midlevel_resnet(x)
        y = self.upsample(y)
        return y


model = Net()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)



