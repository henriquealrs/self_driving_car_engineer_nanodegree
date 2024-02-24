import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import glob

def calculate_mean_std(image_list):
    """
    calculate mean and std of image list
    args:
    - image_list [list[str]]: list of image paths
    returns:
    - mean [array]: 1x3 array of float, channel wise mean
    - std [array]: 1x3 array of float, channel wise std
    """
    mean = np.zeros((len(image_list), 3))
    std = np.zeros((len(image_list), 3))
    for i, filename in enumerate(image_list):
        img = np.array(Image.open(filename).convert('RGB'))
        r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        mean[i, :] = np.mean(r), np.mean(g), np.mean(b)
        std[i, :] = np.std(r), np.std(g), np.std(b)
    mean = np.mean(mean, 0)
    std = np.mean(std, 0)
    print(mean)
    return mean, std


def channel_histogram(image_list):
    """
    calculate channel wise pixel value
    args:
    - image_list [list[str]]: list of image paths
    """
    # IMPLEMENT THIS FUNCTION


if __name__ == "__main__": 
    image_list = glob.glob('data/images/*')
    mean, std = calculate_mean_std(image_list)