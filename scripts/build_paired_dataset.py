import os
from tqdm import tqdm
import torch
import numpy as np
from skimage import io
from skimage import morphology

from stainaire.utils.fdlutils import quick_imshow, quick_load, quick_save

# ${TRAINING_DATASET_ROOT_FOLDER}
# └───images
#     └───0001.jpg
#     └───0002.jpg
#     └───0003.jpg
#     ...
# └───seg_maps
#     └───0001.png
#     └───0002.png
#     └───0003.png
#     ...
# └───edge_maps
#     └───0001.png
#     └───0002.png
#     └───0003.png
#     ...

ds_in = '/mnt/cloudNAS3/fangda/PanNuke/fold3/'
ds_out = '/mnt/cloudNAS3/fangda/HnE_Paired/PanNuke_Fold3/'

os.makedirs(ds_out + 'images', exist_ok=True)
os.makedirs(ds_out + 'seg_maps', exist_ok=True)
os.makedirs(ds_out + 'edge_maps', exist_ok=True)

# Iterate through the images
name_list = [n.split('.')[0] for n in os.listdir(
    ds_in + 'Images') if n.endswith(('png'))]

# edge = io.imread('projects/spade/test_data/cocostuff_test/edge_maps/000000044195.png')
# seg = io.imread('projects/spade/test_data/cocostuff_test/seg_maps/000000044195.png')

for fname in tqdm(name_list):
    image = io.imread(os.path.join(ds_in, 'Images', f'{fname}.png'))
    label = quick_load(os.path.join(ds_in, 'Labels', f'{fname}.npz'))

    # Image
    image = image
    io.imsave(os.path.join(ds_out, 'images', f'{fname}.jpg'), image)

    # Semantic seg map
    seg = label[1].astype('uint8')
    io.imsave(os.path.join(ds_out, 'seg_maps', f'{fname}.png'), seg)

    # Edge map
    edge = label[0] - morphology.erosion(label[0])
    edge[edge > 0] = 255
    edge = edge.astype('uint8')
    io.imsave(os.path.join(ds_out, 'edge_maps', f'{fname}.png'), edge)
