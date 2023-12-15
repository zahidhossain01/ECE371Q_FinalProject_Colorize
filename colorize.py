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
    color_image = np.copy(LAB)
    color_image[:,:,0] = color_image[:,:,0] * (100/255)
    color_image[:,:,1:3] = color_image[:,:,1:3]
    # color_image[:,:,0] = color_image[:,:,0] * 100
    # color_image[:,:,1:3] = color_image[:,:,1:3] * 255 - 128
    RGB = lab2rgb(color_image.astype(np.float64))
    return RGB, LAB


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
    RGB, LAB = merge_lab_to_rgb(L_input.cpu().squeeze(0), AB_output.detach().cpu().squeeze(0))

    height_diff = 608 - original_shape[0]
    width_diff = 608 - original_shape[1]

    crop_top = height_diff // 2
    crop_left = width_diff // 2
    crop_bottom = crop_top + original_shape[0]
    crop_right = crop_left + original_shape[1]

    RGB = RGB[crop_top:crop_bottom, crop_left:crop_right]
    LAB = LAB[crop_top:crop_bottom, crop_left:crop_right]

    return RGB, LAB

if __name__ == "__main__":
    img_path = "datasets\\source_images_compressed\\canada_20190809_141717.jpg"
    RGB, LAB = colorize(img_path)
    print()

    print(f"RGB Array: {RGB.shape}, {RGB.dtype}")
    print(f"LAB Array: {LAB.shape}, {LAB.dtype}")
    print()

    print(f"R | min: {np.min(RGB[:,:,0])}, max: {np.max(RGB[:,:,0])}")
    print(f"G | min: {np.min(RGB[:,:,1])}, max: {np.max(RGB[:,:,1])}")
    print(f"B | min: {np.min(RGB[:,:,2])}, max: {np.max(RGB[:,:,2])}")
    print()

    print(f"L | min: {np.min(LAB[:,:,0])}, max: {np.max(LAB[:,:,0])}")
    print(f"A | min: {np.min(LAB[:,:,1])}, max: {np.max(LAB[:,:,1])}")
    print(f"B | min: {np.min(LAB[:,:,2])}, max: {np.max(LAB[:,:,2])}")
    print()

    fig, axes = plt.subplots(2,4)
    
    axes[1,0].imshow(RGB[:,:,0], vmin=0, vmax=1, cmap='Reds')
    axes[1,1].imshow(RGB[:,:,1], vmin=0, vmax=1, cmap='Greens')
    axes[1,2].imshow(RGB[:,:,2], vmin=0, vmax=1, cmap='Blues')
    axes[1,3].imshow(RGB)
    axes[1,0].set_title("R")
    axes[1,1].set_title("G")
    axes[1,2].set_title("B")
    axes[1,3].set_title("RGB")

    axes[0,0].imshow(LAB[:,:,0], cmap='gray')
    axes[0,1].imshow(LAB[:,:,1], cmap='gray')
    axes[0,2].imshow(LAB[:,:,2], cmap='gray')
    axes[0,0].set_title("L")
    axes[0,1].set_title("A")
    axes[0,2].set_title("B")

    for ax in axes.flat:
        ax.axis('off')
    fig.tight_layout()
    plt.show()
    