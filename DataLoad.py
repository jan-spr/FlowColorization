import os
import numpy as np

import torch
from torch.utils.data import Dataset

import cv2
from cv2 import imread
from cv2 import cvtColor

# Contains functions for loading data from the DAVIS dataset:
# 1. CustomImageDataset: pytorch dataset class for loading images from the DAVIS dataset
# 2. images_to_tensor, images_normalize, input_concat, target_concat: helper functions

def get_all_image_paths(dataset_path, subdir, res, method):
    col_path = os.path.join(dataset_path, subdir, res)
    gray_path = os.path.join(dataset_path, subdir, res + '_gray')

    if method == 'deepflow':
        flow_path = os.path.join(dataset_path, subdir, 'flow', res + '_deepflow')

    return col_path, gray_path, flow_path


def images_to_tensor(sample):
    # Convert images to tensor format for NN input
    col_img, col_prev_img, gray_img, flow_img = sample['col_img'], sample['col_prev_img'], sample['gray_img'], sample['flow_img']
    col_img = cvtColor(col_img, cv2.COLOR_RGB2LAB)
    col_prev_img = cvtColor(col_prev_img, cv2.COLOR_RGB2LAB)
    gray_img = cvtColor(gray_img, cv2.COLOR_RGB2LAB)

    col_img = torch.from_numpy(col_img)
    col_prev_img = torch.from_numpy(col_prev_img)
    gray_img = torch.from_numpy(gray_img)
    if flow_img is not None:
        flow_img = torch.from_numpy(flow_img)

    return col_img, col_prev_img, gray_img, flow_img


def images_normalize(col_img, col_prev_img, gray_img, flow_img):
    # Normalize images to [0,1] for NN
    # flow values divided by 40 to be vaguely whithin +- 1
    col_img = torch.mul(col_img, 1/255)
    col_prev_img = torch.mul(col_prev_img, 1/255)
    gray_img = torch.mul(gray_img, 1/255)
    if flow_img is not None:
        flow_img = torch.mul(flow_img, 1/40)

    return col_img, col_prev_img, gray_img, flow_img

def input_concat(col_prev_img, gray_img, flow_img):
    # Concatenate images to form NN input
    if flow_img is not None:
        input_tens = torch.cat((
            torch.unsqueeze(col_prev_img[...,0], dim=2), # L channel
            torch.unsqueeze(col_prev_img[...,1], dim=2), # A channel
            torch.unsqueeze(col_prev_img[...,2], dim=2), # B channel
            torch.unsqueeze(flow_img[...,0], dim=2),    # x flow
            torch.unsqueeze(flow_img[...,1], dim=2),    # y flow
            torch.unsqueeze(gray_img[...,0], dim=2)     # L channel
            ),dim=2)
    else:
        input_tens = torch.cat((
            torch.unsqueeze(col_prev_img[...,0], dim=2), # L channel
            torch.unsqueeze(col_prev_img[...,1], dim=2), # A channel
            torch.unsqueeze(col_prev_img[...,2], dim=2), # B channel
            torch.unsqueeze(gray_img[...,0], dim=2)     # L channel
            ),dim=2)
    return input_tens

def target_concat(col_img):
    # slice LAB color image to get AB channels for NN target/ground truth
    target_tens = col_img[...,1:]
    return target_tens

class CustomImageDataset(Dataset):
    # https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    # pytorch dataset class for loading images from the DAVIS dataset
    def __init__(self, dataset_path, subdir, res, method, use_flow=False, train=True):
        self.use_flow = use_flow
        self.resolution = res
        self.col_dir, self.gray_dir, self.flow_dir = get_all_image_paths(dataset_path, subdir, res, method)
        self.vid_labels = os.listdir(self.col_dir)
        self.vid_labels.sort()
        print('vid labels: {}'.format(self.vid_labels))

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
    
    def __input_concat__(self, col_prev_img, gray_img, flow_img):
        # Concatenate images to form NN input
        if self.use_flow:
            input_tens = torch.cat((
                torch.unsqueeze(col_prev_img[...,0], dim=2), # L channel
                torch.unsqueeze(col_prev_img[...,1], dim=2), # A channel
                torch.unsqueeze(col_prev_img[...,2], dim=2), # B channel
                torch.unsqueeze(flow_img[...,0], dim=2),    # x flow
                torch.unsqueeze(flow_img[...,1], dim=2),    # y flow
                torch.unsqueeze(gray_img[...,0], dim=2)     # L channel
                ),dim=2)
        else:
            input_tens = torch.cat((
                torch.unsqueeze(col_prev_img[...,0], dim=2), # L channel
                torch.unsqueeze(col_prev_img[...,1], dim=2), # A channel
                torch.unsqueeze(col_prev_img[...,2], dim=2), # B channel
                torch.unsqueeze(gray_img[...,0], dim=2)     # L channel
                ),dim=2)
        return input_tens

    def __target_concat__(self, col_img):
        # slice LAB color image to get AB channels for NN target/ground truth
        target_tens = col_img[...,1:]
        return target_tens

    def __getitem__(self, idx, verbose=False):
        # get item from dataset for NN training
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

        col_img, col_prev_img, gray_img, flow_img = images_to_tensor(sample)
        col_img, col_prev_img, gray_img, flow_img = images_normalize(col_img, col_prev_img, gray_img, flow_img)
        input = self.__input_concat__(col_prev_img, gray_img, flow_img)
        output = self.__target_concat__(col_img)
        input = torch.transpose(input, 0, 2)
        output = torch.transpose(output, 0, 2)

        return input, output
    

def data_to_images(input, output, use_flow=False, input_only=False):
    # create viewable rgb images from NN input and output tensors
    # use_flow: if true, input has 6 channels, else 4
    # input_only: if true, only return input images

    input = torch.transpose(input, 0, 2)
    output = torch.transpose(output, 0, 2)

    if use_flow:
        col_img = input.numpy()[:, :, 0:3]
        flow_img = input.numpy()[:, :, 3:5]
        grey_img = input.numpy()[:, :, 5:6]
    else:
        col_img = input.numpy()[:, :, 0:3]
        grey_img = input.numpy()[:, :, 3:4]
    
    col_img  = (col_img*255).astype(np.uint8)
    col_image = cv2.cvtColor(col_img, cv2.COLOR_LAB2RGB)

    if not input_only:
        output = output.numpy()
        output = (output*255).astype(np.uint8)
        lab = np.zeros((grey_img.shape[0], grey_img.shape[1], 3))
        lab[:,:,0] = grey_img[:,:,0]*255
        lab[:,:,1] = output[:,:,0]
        lab[:,:,2] = output[:,:,1]
        lab = lab.astype(np.uint8)
        pred_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    if input_only:
        if use_flow:
            return col_image, flow_img, grey_img
        else:
            return col_image, None, grey_img
    else:
        if use_flow:
            return col_image, flow_img, grey_img, pred_rgb
        else:
            return col_image, None, grey_img, pred_rgb

def images_to_data(col_img, flow_img, gray_img):
    sample = {
            'col_img': col_img,
            'col_prev_img': col_img,
            'gray_img': gray_img,
            'flow_img': flow_img
        }

    col_img, col_prev_img, gray_img, flow_img = images_to_tensor(sample)
    col_img, col_prev_img, gray_img, flow_img = images_normalize(col_img, col_prev_img, gray_img, flow_img)
    input = input_concat(col_prev_img, gray_img, flow_img)
    output = target_concat(col_img)
    input = torch.transpose(input, 0, 2)
    output = torch.transpose(output, 0, 2)

    return input, output