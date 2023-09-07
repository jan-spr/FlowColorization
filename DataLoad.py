import os

import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor


import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


dataset_path = '~/Documents/Colorization/Datasets/'
dataset_name = 'DAVIS'
image_folder = 'JPEGImages'
nd_array_folder = 'nd_arrays'
dataset_path = os.path.expanduser(dataset_path)
dataset_path = os.path.join(dataset_path, dataset_name)
img_path = os.path.join(dataset_path, image_folder)
nd_array_path = os.path.join(dataset_path, nd_array_folder)


class CustomImageDataset(Dataset):
    # https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    def __init__(self, resolution):
        # Load colored images
        self.col_dir = os.path.join(img_path, resolution)
        
        self.vid_labels = os.listdir(self.col_dir)
        self.vid_labels.sort()
        print('vid labels: {}'.format(self.vid_labels))

        # Load gray images
        self.gray_dir = os.path.join(img_path, resolution+'_gray')

        # Load flow images
        # nd_arrays are stored in a different folder
        self.flow_dir = os.path.join(nd_array_path, resolution+'_deepflow')

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
        return length

    def __idx_to_vid__(self, idx):
        for vid in self.vid_labels:
            if idx < self.vid_lengths[vid]:
                return vid, idx
            else:
                idx -= self.vid_lengths[vid]

    def __getitem__(self, idx):
        vid_label, frame_idx = self.__idx_to_vid__(idx)
        frame_idx += 1

        col_dir = os.path.join(self.col_dir, vid_label)
        gray_dir = os.path.join(self.gray_dir, vid_label)
        flow_dir = os.path.join(self.flow_dir, vid_label)

        col_frame = os.path.join(col_dir, str(frame_idx).zfill(5) + '.jpg')
        gray_frame = os.path.join(gray_dir, str(frame_idx).zfill(5) + '.jpg')
        flow_frame = os.path.join(flow_dir, str(frame_idx).zfill(5) + '.jpg.npy')

        print('col_frame: {}'.format(col_frame))
        print('gray_frame: {}'.format(gray_frame))
        print('flow_frame: {}'.format(flow_frame))

        col_img = read_image(col_frame)
        gray_img = read_image(gray_frame)
        flow_img = np.load(flow_frame)

        sample = {'col_img': col_img, 'gray_img': gray_img, 'flow_img': flow_img}

        return sample, vid_label, frame_idx