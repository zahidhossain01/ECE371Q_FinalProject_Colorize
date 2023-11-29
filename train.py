import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import torchvision.models as models
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
import os

# CUDA stuff
# https://stackoverflow.com/questions/53695105/why-we-need-image-tocuda-when-we-have-model-tocuda
# https://pytorch.org/docs/stable/notes/cuda.html#cuda-semantics
# either tensor.cuda() or tensor.to(device=cuda) (where cuda=torch.device('cuda'))
BATCH_SIZE = 100

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = os.listdir(root_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = PIL.Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        return image


class SquarePadImage():
    def __init__(self, sidelength):
        self.sidelength = sidelength
    def __call__(self, img:PIL.Image.Image):
        I = np.asarray(img)
        height_diff = self.sidelength - I.shape[0]
        width_diff = self.sidelength - I.shape[1]
        pad_up = height_diff // 2
        pad_down = height_diff - pad_up
        pad_left = width_diff // 2
        pad_right = width_diff - pad_left
        I_pad = np.pad(I, ((pad_up, pad_down), (pad_left, pad_right)), constant_values=0)
        return PIL.Image.fromarray(I_pad)

transform = transforms.Compose([
    SquarePadImage(2000),
    transforms.ToTensor(),
])
train_set = ImageDataset(root_dir="datasets/training/bw", transform=transform)
val_set = ImageDataset(root_dir="datasets/validation/bw", transform=transform)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)


examples = enumerate(train_loader)
batch_idx, images = next(examples)
print(f"Batch #{batch_idx} | {len(images)} images of type {type(images[0])}")
batch_idx, images = next(examples)
print(f"Batch #{batch_idx} | {len(images)} images of type {type(images[0])}")
batch_idx, images = next(examples)
print(f"Batch #{batch_idx} | {len(images)} images of type {type(images[0])}")

read_img = to_pil_image(images[0])
print(read_img)
plt.imshow(read_img, cmap='gray', vmin=0, vmax=255)
plt.show()

# resnet = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
# for c in resnet.children():
#     print(c)
#     print()
