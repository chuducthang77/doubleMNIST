import numpy as np
import matplotlib.pyplot as plt
import supervision as sv

# Load the segmentation and label file
train_seg = np.load('./train_seg.npy')
train_seg = train_seg.reshape(-1, 1, 64*64)
train_seg = np.repeat(train_seg, 2, axis=1)

train_Y = np.load('./train_Y.npy')
train_Y = train_Y.reshape(-1, 2, 1)

# Create a mask (binary) for each label and each image
train_mask = (train_seg == train_Y).reshape(-1, 2, 64, 64)

# Load the segmentation and label file
valid_seg = np.load('./valid_seg.npy')
valid_seg = valid_seg.reshape(-1, 1, 64*64)
valid_seg = np.repeat(valid_seg, 2, axis = 1)

valid_Y = np.load('./valid_Y.npy')
valid_Y = valid_Y.reshape(-1, 2, 1)

# Create a mask (binary) for each label and each image
valid_mask = (valid_seg == valid_Y).reshape(-1, 2, 64, 64)


# Test the method
# unique, counts = np.unique(valid_seg[2], return_counts=True)
# print(unique) # Different polygon length
# print(counts)
# _, counts = np.unique(valid_mask[2][0], return_counts=True)
# print(counts)

modified_train_Y = np.empty((train_Y.shape[0], 2), dtype=object)
for i in range(len(train_mask)):
    arr = []
    for j in range(len(train_mask[i])):
        # Convert from mask to polygons
        mask = sv.mask_to_polygons(train_mask[i][j])

        # Normalize the polygon coordinates
        mask = mask[0].flatten() / 64

        # Save the result for each image
        arr.append(np.concatenate([train_Y[i][j] , mask]).tolist())
        modified_train_Y[i, j] = np.concatenate([train_Y[i][j] , mask])

    # Uncoment if we train local GPU
    # with open(f'../datasets/custom-seg/labels/valid/{i}.txt', 'w') as file:
    #     for row in arr:
    #         file.write(' '.join([str(item) for item in row]))
    #         file.write('\n')


modified_valid_Y = np.empty((valid_Y.shape[0], 2), dtype=object)
for i in range(len(valid_mask)):
    arr = []
    for j in range(len(valid_mask[i])):
        # Convert from mask to polygons
        mask = sv.mask_to_polygons(valid_mask[i][j])

        # Normalize the polygon coordinates
        mask = mask[0].flatten() / 64

        # Save the result for each image
        arr.append(np.concatenate([valid_Y[i][j] , mask]).tolist())
        modified_valid_Y[i, j] = np.concatenate([valid_Y[i][j] , mask])

    # Uncoment if we train local GPU
    # with open(f'../datasets/custom-seg/labels/valid/{i}.txt', 'w') as file:
    #     for row in arr:
    #         file.write(' '.join([str(item) for item in row]))
    #         file.write('\n')


np.save('m_train_Y_seg.npy', np.array(modified_train_Y))
np.save('m_valid_Y_seg.npy', np.array(modified_valid_Y))