import numpy as np
import os
from PIL import Image

# Load the train set
train_X = np.load("./train_X.npy")
train_Y = np.load("./train_Y.npy")
train_bboxes = np.load("./train_bboxes.npy")
modified_train_bboxes = np.copy(train_bboxes)

# Change the format from (y_min, x_min, y_max, x_max) to (x_center, y_center, width, height)
modified_train_bboxes[:, :, 0] = (train_bboxes[:, :, 1] + (train_bboxes[:, :, 3] - train_bboxes[:, :, 1])/2)
modified_train_bboxes[:, :, 1] = (train_bboxes[:, :, 0] + (train_bboxes[:, :, 2] - train_bboxes[:, :, 0])/2)
modified_train_bboxes[:, :, 2] = (train_bboxes[:, :, 3] - train_bboxes[:, :, 1])
modified_train_bboxes[:, :, 3] = (train_bboxes[:, :, 2] - train_bboxes[:, :, 0])

# Change the scale to 0-1
modified_train_bboxes = modified_train_bboxes / 64

# Concatenate the label
train_Y = train_Y.reshape(train_Y.shape[0], train_Y.shape[1], 1)
train_Y = np.concatenate((train_Y, modified_train_bboxes), axis=2)

# Save the img as well as the label
for i in range(len(train_Y)):
    im = Image.fromarray(train_X[i].reshape(64, 64, 3))
    im.save(f"../datasets/custom/images/train/{i}.jpeg")
    np.savetxt(f'../datasets/custom/labels/train/{i}.txt', train_Y[i])

# Save the numpy file
np.save('m_train_Y.npy',train_Y)


# Load the validation file
valid_X = np.load("./valid_X.npy")
valid_Y = np.load("./valid_Y.npy")
valid_bboxes = np.load("./valid_bboxes.npy")
modified_valid_bboxes = np.copy(valid_bboxes)

# Change the format from (y_min, x_min, y_max, x_max) to (x_center, y_center, width, height)
modified_valid_bboxes[:, :, 0] = (valid_bboxes[:, :, 1] + (valid_bboxes[:, :, 3] - valid_bboxes[:, :, 1])/2)
modified_valid_bboxes[:, :, 1] = (valid_bboxes[:, :, 0] + (valid_bboxes[:, :, 2] - valid_bboxes[:, :, 0])/2)
modified_valid_bboxes[:, :, 2] = (valid_bboxes[:, :, 3] - valid_bboxes[:, :, 1])
modified_valid_bboxes[:, :, 3] = (valid_bboxes[:, :, 2] - valid_bboxes[:, :, 0])

# Change the scale to 0-1
modified_valid_bboxes = modified_valid_bboxes / 64

# Concatenate the label
valid_Y = valid_Y.reshape(valid_Y.shape[0], valid_Y.shape[1], 1)
valid_Y = np.concatenate((valid_Y, modified_valid_bboxes), axis=2)

# Save the img as well as the label
for i in range(len(valid_Y)):
    im = Image.fromarray(valid_X[i].reshape(64, 64, 3))
    im.save(f"../datasets/custom/images/valid/{i}.jpeg")
    np.savetxt(f'../datasets/custom/labels/valid/{i}.txt', valid_Y[i])

# Save the numpy file
np.save('m_valid_Y.npy',valid_Y)
