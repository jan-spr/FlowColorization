import os

import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

import cv2
from cv2 import imread
from cv2 import cvtColor


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
    def __init__(self, resolution, use_flow=False):
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
        raise IndexError('Index out of range')
    
    def __images_to_tensor__(self, sample):
        col_img, col_prev_img, gray_img, flow_img = sample['col_img'], sample['col_prev_img'], sample['gray_img'], sample['flow_img']
        col_img = cvtColor(col_img, cv2.COLOR_RGB2LAB)
        col_prev_img = cvtColor(col_prev_img, cv2.COLOR_RGB2LAB)
        gray_img = cvtColor(gray_img, cv2.COLOR_RGB2LAB)
    
        col_img = torch.from_numpy(col_img)
        col_prev_img = torch.from_numpy(col_prev_img)
        gray_img = torch.from_numpy(gray_img)
        flow_img = torch.from_numpy(flow_img)

        return col_img, col_prev_img, gray_img, flow_img

    def __images_normalize__(self, col_img, col_prev_img, gray_img, flow_img):
        # Normalize images to [0,1] for NN
        # flow values vaguely around +- 1
        col_img = torch.mul(col_img, 1/255)
        col_prev_img = torch.mul(col_prev_img, 1/255)
        gray_img = torch.mul(gray_img, 1/255)
        flow_img = torch.mul(flow_img, 1/10)

        return col_img, col_prev_img, gray_img, flow_img
    
    def __input_concat__(self, col_prev_img, gray_img, flow_img):
    
        input_tens = torch.cat((
            torch.unsqueeze(col_prev_img[...,0], dim=2), # L channel
            torch.unsqueeze(col_prev_img[...,1], dim=2), # A channel
            torch.unsqueeze(col_prev_img[...,2], dim=2), # B channel
            torch.unsqueeze(flow_img[...,0], dim=2),    # x flow
            torch.unsqueeze(flow_img[...,1], dim=2),    # y flow
            torch.unsqueeze(gray_img[...,0], dim=2)     # L channel
            ),dim=2)
        return input_tens

    def __target_concat__(self, col_img):
        target_tens = col_img[...,1:]
        return target_tens

    def __getitem__(self, idx, verbose=False):
        vid_label, frame_idx = self.__idx_to_vid__(idx)
        frame_idx += 1

        col_dir = os.path.join(self.col_dir, vid_label)
        gray_dir = os.path.join(self.gray_dir, vid_label)
        flow_dir = os.path.join(self.flow_dir, vid_label)

        col_frame = os.path.join(col_dir, str(frame_idx).zfill(5) + '.jpg')
        col_prev_frame = os.path.join(col_dir, str(frame_idx-1).zfill(5) + '.jpg')
        gray_frame = os.path.join(gray_dir, str(frame_idx).zfill(5) + '.jpg')
        flow_frame = os.path.join(flow_dir, str(frame_idx).zfill(5) + '.jpg.npy')
        
        if verbose:
            print('col_frame: {}'.format(col_frame))
            print('gray_frame: {}'.format(gray_frame))
            print('flow_frame: {}'.format(flow_frame))

        col_img = imread(col_frame)
        col_prev_img = imread(col_prev_frame)
        gray_img = imread(gray_frame)
        flow_img = np.load(flow_frame)

        sample = {
            'col_img': col_img,
            'col_prev_img': col_prev_img,
            'gray_img': gray_img,
            'flow_img': flow_img
        }

        col_img, col_prev_img, gray_img, flow_img = self.__images_to_tensor__(sample)

        col_img, col_prev_img, gray_img, flow_img = self.__images_normalize__(col_img, col_prev_img, gray_img, flow_img)
        
        input = self.__input_concat__(col_prev_img, gray_img, flow_img)

        output = self.__target_concat__(col_img)

        return input, output
    

