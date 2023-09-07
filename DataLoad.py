import os

import torch
from torchvision.io import read_image

dataset_path = '~/Documents/Colorization/Datasets/'
dataset_name = 'DAVIS'
image_folder = 'JPEGImages'
dataset_path = os.path.expanduser(dataset_path)
dataset_path = os.path.join(dataset_path, dataset_name)
img_path = os.path.join(dataset_path, image_folder)



class CustomImageDataset(Dataset):
    def __init__(self, resolution, transform=None, target_transform=None):
        self.col_dir = os.path.join(img_path, resolution)
        self.vid_labels = os.listdir(self.col_dir)

        self.gray_dir = os.path.join(img_path, resolution+'_gray')
        self.flow_dir = os.path.join(img_path, resolution+'_deepflow')

        self.transform = transform
        self.target_transform = target_transform

        # dictionary of video names and their lengths
        self.vid_lengths = {}
        for vid in self.vid_labels:
            col_dir = os.path.join(self.col_dir, vid)
            gray_dir = os.path.join(self.gray_dir, vid)
            flow_dir = os.path.join(self.flow_dir, vid)

            col_frames = os.listdir(col_dir)
            gray_frames = os.listdir(gray_dir)
            flow_frames = os.listdir(flow_dir)

            assert len(col_frames) == len(gray_frames) == len(flow_frames) + 1
            self.vid_lengths[vid] = len(flow_frames)

    def __len__(self):
        length = 0
        for vid in self.vid_labels:
            length += self.vid_lengths[vid]
        

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
