import numpy as np
import cv2
import os
import skimage
import torch

# just a collection of functions, to reduce clutter in the notebooks
# e.g. to convert between different image formats, or create video from images

def make_dir(path):
    # Creates a directory if it doesn't exist already
    if not os.path.exists(path):
        os.makedirs(path)

def cartToPol(x, y):  
    # Convert cartesian to polar coordinates (for optical flow)
    ang = np.arctan2(y, x)
    mag = np.hypot(x, y)
    return mag, ang

def uv_2_rgb(image_uv, resize=False):
    # Convert the optical flow field into HSV Polar coordinate representation

    uv_shape = image_uv.shape
    hsv = np.zeros((uv_shape[0], uv_shape[1], 3))
    hsv[..., 1] = 255

    # Encoding: convert the algorithm's output into Polar coordinates
    mag, ang = cartToPol(image_uv[..., 0], image_uv[..., 1])
    # Use Hue and Value to encode the Optical Flow
    hsv[..., 0] = (ang+np.pi) * 180 / ( 2 * np.pi )
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    #print(hsv)
    hsv = np.round(hsv).astype(np.uint8)
    #print(hsv)
    # Convert HSV to RGB (BGR) color representation
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def images_2_video(image_folder, video_name, fps=30):
    """
    Takes the images of a folder and form them into a video.

    :param image_folder: the folder where the images are stores as a string of the path
    :param video_name: the name of the video as a string
    :param fps: the number of fps as int
    """

    image_names = os.listdir(image_folder)
    image_names.sort()
    frame = cv2.imread(os.path.join(image_folder, image_names[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, fps, (width,height))

    for image in image_names:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

def all_images_2_video(dataset_path, subdir, res, video_folder, fps):
    """
    Creates videos of all subdirectories of the dataset_path.

    :param dataset_path: the path to the (DAVIS) dataset as a string
    :param subdir: the name of the subdirectory of DAVIS as a string. whether 'train', 'val' or 'test'
    :param res: the resolution of the images as string, e.g. '176p'
    :param video_folder: the name of the folder where the videos should be stored as a string
    :param fps: the number of fps as int
    """

    image_path = os.path.join(dataset_path, subdir, res)
    video_path = os.path.join(dataset_path, subdir, video_folder, res)

    make_dir(video_path)

    print('Image path: {}'.format(image_path))
    print('Video path: {}'.format(video_path))

    subdirs = os.listdir(image_path)
    print('Number of subdirectories: {}'.format(len(subdirs)))

    for subdir in tqdm(subdirs):
        if subdir == '.DS_Store': continue
        image_folder = os.path.join(image_path, subdir)
        video_name = os.path.join(video_path, subdir + '.avi')
        images_2_video(image_folder, video_name, fps)

def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

def save_image(image, image_path):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(image_path, image)

def dense_optical_flow(input_image_1, input_image_2, method="farneback", params=[], to_gray=True):
    # get the dense optical flow field between two images
    # input_image_1: first image
    # input_image_2: second image
    # method: optical flow method (farneback, deepflow)

    if method == "farneback":
        method = cv2.calcOpticalFlowFarneback
        # params: pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags
        params = [0.5, 3, 15, 3, 5, 1.2, 0]  # Farneback's algorithm parameters
    elif method == "deepflow":
        deepflow = cv2.optflow.createOptFlow_DeepFlow()
        method = deepflow.calc
        params = []

    # create HSV & make Saturation a constant
    hsv = np.zeros_like(input_image_1)
    hsv[..., 1] = 255

    # Convert to grayscale
    if to_gray:
        input_image_1 = cv2.cvtColor(input_image_1, cv2.COLOR_BGR2GRAY)
        input_image_2 = cv2.cvtColor(input_image_2, cv2.COLOR_BGR2GRAY)

    # Calculate dense optical flow by Farneback method
    flow = method(input_image_1, input_image_2, None, *params)
    return flow

def resize_image(input_image, target_res = (176,320), L_img=None):
    # resize image to target resolution
    # input_image: image to resize
    # target_res: target resolution
    res_img = skimage.transform.resize(input_image, target_res, preserve_range=True)

    if L_img is not None:
        res_img = cv2.cvtColor(res_img, cv2.COLOR_RGB2LAB)
        L_img = cv2.cvtColor(L_img, cv2.COLOR_RGB2LAB)
        res_img[:,:,0] = L_img[:,:,0]
        res_img = cv2.cvtColor(res_img, cv2.COLOR_LAB2RGB)

    return res_img.astype(np.uint8)

def checkpoint(folder, model, filename):
    # Creates a checkpoint of the model
    filename = os.path.join(folder, filename)
    torch.save(model.state_dict(), filename)


def resume(folder, model, filename):
    # Loads a current (already trained) model.
    filename = os.path.join(folder, filename)
    model.load_state_dict(torch.load(filename))