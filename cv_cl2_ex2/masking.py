from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np

def create_mask(path, color_threshold):
    """
    create a binary mask of an image using a color threshold
    args:
    - path [str]: path to image file
    - color_threshold [array]: 1x3 array of RGB value
    returns:
    - img [array]: RGB image array
    - mask [array]: binary array
    """
    img = np.array(Image.open(path))
    print(img.shape)
    # print(img[:,:] > color_threshold)
    mask_colors = (img[:,:] > color_threshold)
    mask = np.logical_and(mask_colors[:,:,0], mask_colors[:,:,1], mask_colors[:,:,2])
    print(mask.shape)
    return img, mask


def mask_and_display(img: np.ndarray, mask):
    """
    display 3 plots next to each other: image, mask and masked image
    args:
    - img [array]: HxWxC image array
    - mask [array]: HxW mask array
    """
    img_masked = np.zeros(img.shape)
    img_masked[:, :, 0] = img[:,:,0] * mask
    img_masked[:, :, 1] = img[:,:,1] * mask
    img_masked[:, :, 2] = img[:,:,2] * mask
    img_masked = img_masked.astype(int)
    print("Unmasked")
    print(img[int(1280/2-3):int(1280/2+3), int(1920/2-3):int(1920/2+3), :])
    print("Masked")
    print(img_masked[int(1280/2-3):int(1280/2+3), int(1920/2-3):int(1920/2+3), :])
    print(img_masked.shape)
    print(f"img {img.dtype} - img_masked: {img_masked.dtype}")
    plt.imshow(img_masked)
    plt.show()


if __name__ == '__main__':
    path = 'data/images/segment-1231623110026745648_480_000_500_000_with_camera_labels_38.png'
    color_threshold = [128, 128, 128]
    img, mask = create_mask(path, color_threshold)
    mask_and_display(img, mask)