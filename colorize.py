import torch
import torch.nn as nn
import torchvision.models as models
import PIL.Image
import PIL.ImageOps
import numpy as np
from skimage.color import lab2rgb, rgb2gray, rgb2lab
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
    intermediate = np.copy(LAB).astype(np.float64)
    intermediate[:,:,0] = intermediate[:,:,0] * (100/255)
    intermediate[:,:,1:3] = intermediate[:,:,1:3]
    # RGB = lab2rgb(intermediate.astype(np.float64))
    RGB = lab2rgb(intermediate)
    return RGB, LAB, intermediate


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
    RGB, LAB, intermediate = merge_lab_to_rgb(L_input.cpu().squeeze(0), AB_output.detach().cpu().squeeze(0))

    height_diff = 608 - original_shape[0]
    width_diff = 608 - original_shape[1]

    crop_top = height_diff // 2
    crop_left = width_diff // 2
    crop_bottom = crop_top + original_shape[0]
    crop_right = crop_left + original_shape[1]

    RGB = RGB[crop_top:crop_bottom, crop_left:crop_right]
    LAB = LAB[crop_top:crop_bottom, crop_left:crop_right]
    intermediate = intermediate[crop_top:crop_bottom, crop_left:crop_right]

    return RGB, LAB, intermediate

if __name__ == "__main__":
    img_path = "datasets\\source_images_compressed\\canada_20190809_141717.jpg"
    # RGB, LAB, LAB_int = colorize(img_path)
    RGB = np.copy(np.asarray(PIL.Image.open(img_path)))
    LAB = rgb2lab(RGB)
    LAB_int = np.copy(LAB)
    print()

    print(f"LAB Array: {LAB.shape}, {LAB.dtype}")
    print(f"LAB_int Array: {LAB_int.shape}, {LAB_int.dtype}")
    print(f"RGB Array: {RGB.shape}, {RGB.dtype}")
    print()
    print(f"L | min: {np.min(LAB[:,:,0]):.2f}, max: {np.max(LAB[:,:,0]):.2f}")
    print(f"A | min: {np.min(LAB[:,:,1]):.2f}, max: {np.max(LAB[:,:,1]):.2f}")
    print(f"B | min: {np.min(LAB[:,:,2]):.2f}, max: {np.max(LAB[:,:,2]):.2f}")
    print()
    print(f"L*| min: {np.min(LAB_int[:,:,0]):.2f}, max: {np.max(LAB_int[:,:,0]):.2f}")
    print(f"A*| min: {np.min(LAB_int[:,:,1]):.2f}, max: {np.max(LAB_int[:,:,1]):.2f}")
    print(f"B*| min: {np.min(LAB_int[:,:,2]):.2f}, max: {np.max(LAB_int[:,:,2]):.2f}")
    print()
    print(f"R | min: {np.min(RGB[:,:,0]):.2f}, max: {np.max(RGB[:,:,0]):.2f}")
    print(f"G | min: {np.min(RGB[:,:,1]):.2f}, max: {np.max(RGB[:,:,1]):.2f}")
    print(f"B | min: {np.min(RGB[:,:,2]):.2f}, max: {np.max(RGB[:,:,2]):.2f}")
    print()

    fig, axes = plt.subplots(3,4)
    
    # cmaps for LAB:
    # L: black to white (cmap='gray') | generally [0,100]
    # A: green to red | generally [-100,100] or [-128,127]
    # B: blue to yellow | generally [-100,100] or [-128,127]

    axes[0,0].imshow(LAB[:,:,0], cmap='gray')
    axes[0,1].imshow(LAB[:,:,1], cmap='gray')
    axes[0,2].imshow(LAB[:,:,2], cmap='gray')
    axes[0,0].set_title("L")
    axes[0,1].set_title("A")
    axes[0,2].set_title("B")

    axes[1,0].imshow(LAB_int[:,:,0], vmin=0, vmax=100, cmap='gray')
    axes[1,1].imshow(LAB_int[:,:,1], cmap='PiYG_r')
    axes[1,2].imshow(LAB_int[:,:,2], cmap='BrBG_r')
    axes[1,0].set_title("L*")
    axes[1,1].set_title("A*")
    axes[1,2].set_title("B*")

    axes[2,0].imshow(RGB[:,:,0], vmin=0, vmax=1, cmap='Reds')
    axes[2,1].imshow(RGB[:,:,1], vmin=0, vmax=1, cmap='Greens')
    axes[2,2].imshow(RGB[:,:,2], vmin=0, vmax=1, cmap='Blues')
    axes[2,3].imshow(RGB)
    axes[2,0].set_title("R")
    axes[2,1].set_title("G")
    axes[2,2].set_title("B")
    axes[2,3].set_title("RGB")

    for ax in axes.flat:
        ax.axis('off')
    fig.tight_layout()
    plt.show()
    