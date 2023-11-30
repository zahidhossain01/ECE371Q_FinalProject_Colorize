import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
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
        img_name = os.path.join(self.root_dir, self.images[idx])
        img = np.asarray(PIL.Image.open(img_name))
        # img_og = np.copy(np.asarray(PIL.Image.open(img_name)))
        # img_lab = rgb2lab(img_og)
        # img_lab = (img_lab + 128) / 255
        # img_ab = img_lab[:,:,1:3]
        # img_gray = rgb2gray(img_og)

        if self.transform:
            img = self.transform(img)
            # img_ab = self.transform(img_ab)

        return img




transform = transforms.Compose([
    transforms.Resize((500,500)),
    transforms.ToTensor(),
])
train_set = ImageDataset(root_dir="datasets/training/bw", transform=transform)
val_set = ImageDataset(root_dir="datasets/validation/bw", transform=transform)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)


examples = enumerate(val_loader)
batch_idx, images = next(examples)
print(f"Batch #{batch_idx} | {len(images)} images of type {type(images[0])}")
# img_gray, img_ab = images[0]
img_gray = images[0]
read_img = to_pil_image(img_gray)
print(read_img)
plt.imshow(read_img, cmap='gray', vmin=0, vmax=255)
plt.show()





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



