import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image


# Load the image (Change to the appropriate dataset)
images_dir = "../datasets/custom/images/train/0.jpeg"
labels_dir = "../datasets/custom/labels/train/0.txt"

img = np.asarray(Image.open(images_dir))
labels = np.loadtxt(labels_dir)

height, width, _ = img.shape

# Rescale
labels[:, 1] *= width # x_center
labels[:, 3] *= width # width
labels[:, 2] *= height # y_center
labels[:, 4] *= height # height

# Change from the format (x_center, y_center, width, height) to (y_min, x_min, y_max, x_max)
y_min = labels[:, 2] - labels[:, 4] / 2
y_max = labels[:, 2] + labels[:, 4] / 2
x_min = labels[:, 1] - labels[:, 3] / 2
x_max = labels[:, 1] + labels[:, 3] / 2

# Plot
fig, ax = plt.subplots()
ax.imshow(img)
for i in range(len(x_min)):
    rect = patches.Rectangle((x_min[i], y_min[i]), labels[:, 3][i], labels[:, 4][i], edgecolor='r', facecolor='none')
    ax.add_patch(rect)
plt.show()