import torch
import torch.nn as nn
import torchvision.models as models
import PIL.Image
import PIL.ImageOps
import numpy as np
from skimage.color import lab2rgb, rgb2gray
import matplotlib.pyplot as plt
import os

class ColorizeNet(nn.Module):
    def __init__(self, input_size=128):
        super(ColorizeNet, self).__init__()
        MIDLEVEL_FEATURE_SIZE = 128

        ## First half: ResNet
        resnet = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
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

model = ColorizeNet()
model.load_state_dict(torch.load('models/model-epoch-16-losses-0.002.pth'))
model.eval()

use_gpu = torch.cuda.is_available()
if use_gpu:
    model = model.cuda()

def squarepad(img, sidelength=608):
    """square pad 2d array (grayscale image), not exactly same as the func in process_dataset.py"""
    h,w = img.shape
    
    if(sidelength < max(w, h)):
        sidelength = max(w, h)
    height_diff = sidelength - h
    width_diff = sidelength - w
    pad_up = height_diff // 2
    pad_down = height_diff - pad_up
    pad_left = width_diff // 2
    pad_right = width_diff - pad_left

    I_pad = np.pad(img, ((pad_up, pad_down), (pad_left, pad_right)), constant_values=0)

    return I_pad


def merge_lab_to_rgb(L_input, AB_input):
    LAB = torch.cat((L_input, AB_input), 0).numpy()
    LAB = LAB.transpose((1,2,0))
    LAB[:,:,0] = LAB[:,:,0] * 100
    LAB[:,:,1:3] = LAB[:,:,1:3] * 255 - 128
    RGB = lab2rgb(LAB.astype(np.float64))
    return RGB


def colorize(img_path):
    img_input = PIL.Image.open(img_path).convert("L")
    img_r = np.asarray(PIL.ImageOps.contain(img_input, (608, 608)))
    original_shape = img_r.shape
    L = squarepad(img_r)

    L_input = torch.from_numpy(L).unsqueeze(0).float()
    if use_gpu:
        L_input = L_input.cuda()

    L_input = L_input.unsqueeze(0) # https://discuss.pytorch.org/t/valueerror-expected-4d-input-got-3d-input/150585
    AB_output = None
    with torch.no_grad():
        AB_output = model(L_input)

    print(f"L: {L_input.shape}, AB: {AB_output.shape}")
    RGB = merge_lab_to_rgb(L_input.cpu().squeeze(0), AB_output.detach().cpu().squeeze(0))

    height_diff = 608 - original_shape[0]
    width_diff = 608 - original_shape[1]

    crop_top = height_diff // 2
    crop_left = width_diff // 2
    crop_bottom = crop_top + original_shape[0]
    crop_right = crop_left + original_shape[1]

    RGB = RGB[crop_top:crop_bottom, crop_left:crop_right]

    return RGB

if __name__ == "__main__":
    img_path = "datasets\\source_images_compressed\\canada_20190809_141717.jpg"
    RGB = colorize(img_path)
    print(f"RGB Array: {RGB.shape}, {RGB.dtype}")
    # plt.imshow(RGB)
    plt.imshow(RGB)
    plt.show()
    