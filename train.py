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


# img_transform_size = 224
# BATCH_SIZE = 50
# LEARNING_RATE = .01

# For quick testing, 1m/ep
img_transform_size = 128
BATCH_SIZE = 64
LEARNING_RATE = .0001

# Decent Results, 4m/ep
# img_transform_size = 608
# BATCH_SIZE = 15
# LEARNING_RATE = .0001

best_losses = 1e10
epochs = 2

use_gpu = torch.cuda.is_available()

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
    '''from PyTorch ImageNet tutorial''' 
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
    transforms.Resize((img_transform_size,img_transform_size)),
])
train_set = ImageDataset(root_dir="datasets/training/rgb", transform=transform)
val_set = ImageDataset(root_dir="datasets/validation/rgb", transform=transform)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)


# # Testing loader enumeration
# examples = enumerate(train_loader)
# batch_idx, (b1, b2) = next(examples) # b1 and b2 are both arrays (tensors) of length BATCH_SIZE
# print(f"Batch #{batch_idx} Types   | b1:{type(b1)}, b2:{type(b2)}")
# print(f"Batch #{batch_idx} Lengths | len(b1): {len(b1)}, len(b2): {len(b2)}")

# img_gray_tensor, img_ab_tensor = b1[0], b2[0]
# print(f"Tensor Shapes | gray:{img_gray_tensor.shape}, ab:{img_ab_tensor.shape}")
# read_img = to_pil_image(img_gray_tensor)
# print(read_img)
# plt.imshow(read_img, cmap='gray', vmin=0, vmax=255)
# plt.show()


class Net(nn.Module):
    def __init__(self, input_size=128):
        super(Net, self).__init__()
        MIDLEVEL_FEATURE_SIZE = 128

        ## First half: ResNet
        # resnet = models.resnet18(num_classes=365)
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

def validate(val_loader, model, criterion, save_images, epoch):
    model.eval()

    # Prepare value counters and timers
    batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()

    end = time.time()
    already_saved_images = False
    for i, (input_gray, input_ab) in enumerate(val_loader):
        data_time.update(time.time() - end)

        # Use GPU
        if use_gpu: input_gray, input_ab = input_gray.cuda(), input_ab.cuda()

        # Run model and record loss
        output_ab = model(input_gray) # throw away class predictions
        loss = criterion(output_ab, input_ab)
        losses.update(loss.item(), input_gray.size(0))

        # Save images to file
        if save_images and not already_saved_images:
            already_saved_images = True
            # for j in range(min(len(output_ab), 10)): # save at most 5 images
            for j in range(len(output_ab)): # p sure this is batch size
                save_path = {'grayscale': 'outputs/gray/', 'colorized': 'outputs/color/'}
                save_name = f'img-{i * val_loader.batch_size + j}-epoch-{epoch}.jpg'
                to_rgb(input_gray[j].cpu(), ab_input=output_ab[j].detach().cpu(), save_path=save_path, save_name=save_name)

        # Record time to do forward passes and save images
        batch_time.update(time.time() - end)
        end = time.time()

        # Print model accuracy -- in the code below, val refers to both value and validation
        if i % 25 == 0:
            print('Validate: [{0}/{1}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses))

    print('Finished validation.')
    return losses.avg

def train(train_loader, model, criterion, optimizer, epoch):
    print('Starting training epoch {}'.format(epoch))
    model.train()

    # Prepare value counters and timers
    batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()

    end = time.time()
    for i, (input_gray, input_ab) in enumerate(train_loader):
    
        # Use GPU if available
        if use_gpu: input_gray, input_ab = input_gray.cuda(), input_ab.cuda()

        # Record time to load data (above)
        data_time.update(time.time() - end)

        # Run forward pass
        output_ab = model(input_gray) 
        loss = criterion(output_ab, input_ab) 
        losses.update(loss.item(), input_gray.size(0))

        # Compute gradient and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record time to do forward and backward passes
        batch_time.update(time.time() - end)
        end = time.time()

        # Print model accuracy -- in the code below, val refers to value, not validation
        if i % 25 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses)) 

    print('Finished training epoch {}'.format(epoch))

######################################
model = Net()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)


if use_gpu: 
    loss_fn = loss_fn.cuda()
    model = model.cuda()


os.makedirs('outputs/color', exist_ok=True)
os.makedirs('outputs/gray', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)
save_images = True

# 35 epochs = ~50min


# Train model

loss_epoch_data = {"epochs": [], "losses": []}

t1 = time.perf_counter()
for epoch in range(epochs):
    # Train for one epoch, then validate
    train(train_loader, model, loss_fn, optimizer, epoch)
    with torch.no_grad():
        losses = validate(val_loader, model, loss_fn, save_images, epoch)
        loss_epoch_data["epochs"].append(epoch)
        loss_epoch_data['losses'].append(losses)
    # Save checkpoint and replace old best model if current model is better
    if losses < best_losses:
        best_losses = losses
        torch.save(model.state_dict(), 'checkpoints/model-epoch-{}-losses-{:.3f}.pth'.format(epoch+1,losses))
t2 = time.perf_counter()
print()
print(f"Training Time: {t2-t1:.3f} s = {(t2-t1)/60:.3f} m | {((t2-t1)/60)/epochs:.3f} m/ep")

fig, ax = plt.subplots(1,1)
ax.plot(loss_epoch_data['epochs'], loss_epoch_data['losses'])
fig.suptitle("Training Loss vs Epoch")
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
fig.tight_layout()
plt.show()
