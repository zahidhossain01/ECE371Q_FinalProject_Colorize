import numpy as np
import PIL.Image as Image
from skimage.color import rgb2lab, lab2rgb, rgb2gray
import matplotlib.pyplot as plt

img_path = "datasets/training/rgb/AK_20220317_084616.jpg"

img = Image.open(img_path)

I = np.asarray(img)
I_lab = rgb2lab(I)

# plt.figure()
# plt.subplots(1,4)

plt.subplot(1, 4, 1)
plt.imshow(I)

plt.subplot(1, 4, 2)
plt.imshow(I_lab[:,:,0], cmap='gray', vmin=0, vmax=255)

plt.subplot(1, 4, 3)
plt.imshow(I_lab[:,:,1], cmap='gray')

plt.subplot(1, 4, 4)
plt.imshow(I_lab[:,:,2], cmap='gray')

plt.tight_layout()
plt.show()

