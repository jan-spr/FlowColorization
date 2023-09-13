import numpy as np
import cv2
import os

# just a collection of functions, to reduce clutter in the notebooks
# e.g. to convert between different image formats, or create video from images

def make_dir(path):
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

def images_2_video(image_folder,video_name, fps=30):
    image_names = os.listdir(image_folder)
    image_names.sort()
    frame = cv2.imread(os.path.join(image_folder, image_names[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, fps, (width,height))

    for image in image_names:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()