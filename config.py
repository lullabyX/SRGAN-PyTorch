# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import random
from xmlrpc.client import boolean

import numpy as np
import torch
import os
from torch.backends import cudnn

# Random seed to maintain reproducible results
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
# Use GPU for training by default
device = torch.device("cuda", 0)
# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = True
# When evaluating the performance of the SR model, whether to verify only the Y channel image data
only_test_y_channel = True
# Image magnification factor
upscale_factor = 4
# Current configuration parameter method
mode = "train_srresnet"


# Experiment name, easy to save weights and log files
exp_name = ''
while True: 
    print('Enter directory for to save model weights: ')
    exp_name = input()
    if not os.path.exists(exp_name):
        print('Directory Not Found')
        continue;
    break

# save image directory
save_image_dir = 'drive/MyDrive/Thesis/Train/'

while True: 
    print('Enter drive directory to save image:')
    save_image_dir = input()
    if os.path.exists(save_image_dir):
        print(save_image_dir)
        break
    print('Directory not found')

# if mode == "train_srresnet":
#     # Dataset address
#     train_image_dir = "./data/ImageNet/SRGAN/train"
#     valid_image_dir = "./data/ImageNet/SRGAN/valid"
#     test_lr_image_dir = f"./data/Set5/LRbicx{upscale_factor}"
#     test_hr_image_dir = f"./data/Set5/GTmod12"

#     image_size = 96
#     batch_size = 16
#     num_workers = 4

#     # The address to load the pretrained model
#     pretrained_model_path = "./results/pretrained_models/SRResNet_x4-ImageNet-2096ee7f.pth.tar"

#     # Incremental training and migration training
#     resume = ""

#     # Total num epochs
#     epochs = 44

#     # Optimizer parameter
#     model_lr = 1e-4
#     model_betas = (0.9, 0.999)

# How many iterations to print the training result
print_frequency = 500

valid_print_frequency = 10

# if mode == "train_srgan":
    # Dataset address
clean_image_dir = "./train_B"
noisy_image_dir = "./train_A"

# save image dir
while True: 
    print('Enter directory for clean image: ')
    clean_image_dir = input()
    if not os.path.exists(clean_image_dir):
        print('Clean Image Directory Not Found')
        continue;
    break

# Check if noisy images needs to be generated
print('Do you want to create noisy images on the fly?(yes/no): ')
generate_noisy = 'no'
while True:
    generate_noisy = input()
    if generate_noisy == 'yes' or generate_noisy == 'no':
        break


if generate_noisy == 'no':
    while True:
        print('Enter directory for noisy image: ')
        noisy_image_dir = input()
        if not os.path.exists(noisy_image_dir):
            print('Noisy Image Directory Not Found')
            continue;
        break;

# generate artificial noise
print('Do you want to add artificial noise on the fly?(yes/no): ')
generate_art_noise = 'yes'
while True:
    generate_art_noise = input()
    if generate_art_noise == 'yes' or generate_art_noise == 'no':
        break

valid_image_dir = "./data/ImageNet/SRGAN/valid"
test_lr_image_dir = "./test_A/"
test_hr_image_dir = "./test_B/"

image_size = 128
batch_size = 32
num_workers = 4

# The address to load the pretrained model
pretrained_d_model_path = ""
pretrained_g_model_path = ""

# Incremental training and migration training
resume_d = ""
resume_g = ""

# Total num epochs
epochs: int = 20
epochs = int(input('Number of Epochs')) 

# Number of Residual Blocks in Generator
no_res_block = 10

# Feature extraction layer parameter configuration
feature_model_extractor_node = "features.35"
feature_model_normalize_mean = [0.485, 0.456, 0.406]
feature_model_normalize_std = [0.229, 0.224, 0.225]

# Loss function weight
content_weight = 1.0
adversarial_weight = 0.001

# Optimizer parameter
model_lr = 1e-4
model_betas = (0.9, 0.999)

# Dynamically adjust the learning rate policy
lr_scheduler_step_size = epochs // 2
lr_scheduler_gamma = 0.1

# save image location in google drive
# save_image_dir = 'drive/MyDrive/Thesis/TrainedImagev2'

# if mode == "test":
#     # Test data address
#     lr_dir = f"./data/Set5/LRbicx{upscale_factor}"
#     sr_dir = f"./results/test/{exp_name}"
#     hr_dir = f"./data/Set5/GTmod12"

#     model_path = "./results/pretrained_models/SRResNet_x4-ImageNet-2096ee7f.pth.tar"
