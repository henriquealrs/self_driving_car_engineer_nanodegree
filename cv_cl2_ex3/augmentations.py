import copy

import numpy as np 
from PIL import Image
from utils import *
import json


def calculate_iou(gt_bbox, pred_bbox):
    """
    calculate iou 
    args:
    - gt_bbox [array]: 1x4 single gt bbox
    - pred_bbox [array]: 1x4 single pred bbox
    returns:
    - iou [float]: iou between 2 bboxes
    - [xmin, ymin, xmax, ymax]
    """
    xmin = np.max([gt_bbox[0], pred_bbox[0]])
    ymin = np.max([gt_bbox[1], pred_bbox[1]])
    xmax = np.min([gt_bbox[2], pred_bbox[2]])
    ymax = np.min([gt_bbox[3], pred_bbox[3]])
    
    intersection = max(0, xmax - xmin) * max(0, ymax - ymin)
    gt_area = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])
    pred_area = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
    
    union = gt_area + pred_area - intersection
    return intersection / union, [xmin, ymin, xmax, ymax]


def hflip(img: Image.Image, bboxes: list[list[int]]):
    """
    horizontal flip of an image and annotations
    args:
    - img [PIL.Image]: original image
    - bboxes [list[list]]: list of bounding boxes
    return:
    - flipped_img [PIL.Image]: horizontally flipped image
    - flipped_bboxes [list[list]]: horizontally flipped bboxes
    """
    # IMPLEMENT THIS FUNCTION
    # flipped_img = Image.fromarray(np.array(img)[:, ::-1])
    flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    # plt.subplot(1, 2, 1)
    # plt.imshow(img)
    # plt.subplot(1, 2, 2)
    # plt.imshow(flipped_img)
    # plt.show()
    width, _ = img.size
    print(width)
    
    flipped_boxes = [ [box[0], width - box[3], box[2], width - box[1]] for box in bboxes]
    
    return flipped_img, flipped_boxes


def resize(img: Image.Image, boxes: list[list[int]], size: np.ndarray):
    """
    resized image and annotations
    args:
    - img [PIL.Image]: original image
    - boxes [list[list]]: list of bounding boxes
    - size [array]: 1x2 array [width, height]
    returns:
    - resized_img [PIL.Image]: resized image
    - resized_boxes [list[list]]: resized bboxes
    """
    or_w, or_h = img.size
    resized_image = img.resize(size)
    boxes = np.array(boxes)
    hor_factor = size[0]/or_w
    ver_factor = size[1]/or_h
    resized_boxes = copy.copy(boxes)
    resized_boxes[:, 0] = (ver_factor * resized_boxes[:,0])
    resized_boxes[:, 2] = (ver_factor * resized_boxes[:,2])
    resized_boxes[:, 1] = (hor_factor * resized_boxes[:,1])
    resized_boxes[:, 3] = (hor_factor * resized_boxes[:,3])
    return resized_image, resized_boxes.astype(int)


def random_crop(img, boxes, crop_size, min_area=100):
    """
    random cropping of an image and annotations
    args:
    - img [PIL.Image]: original image
    - boxes [list[list]]: list of bounding boxes
    - crop_size [array]: 1x2 array [width, height]
    - min_area [int]: min area of a bbox to be kept in the crop
    returns:
    - cropped_img [PIL.Image]: resized image
    - cropped_boxes [list[list]]: resized bboxes
    """
    # IMPLEMENT THIS FUNCTION
    return cropped_image, cropped_boxes

if __name__ == '__main__':
    # fix seed to check results
    np.random.seed(48)
    
    # open annotations
    with open('data/ground_truth.json', 'r') as f:
        gt_data = json.load(f)
    
    # filter annotations and open image
    filename = 'segment-12208410199966712301_4480_000_4500_000_with_camera_labels_79.png'
    instance = None
    for item in gt_data:
        if item['filename'] == filename:
            instance = item
            break
    if instance is None:
        print("Ground truth not found")
        exit(1)
    print(instance['boxes'])
    boxes = instance['boxes']
    img = Image.open(f"data/images/{filename}")
    print(type(boxes))
    
    # check horizontal flip, resize and random crop
    flp_img, flp_boxes = hflip(img, boxes)
    display_results(img, boxes, flp_img, flp_boxes)
    check_results(flp_img, flp_boxes, 'hflip')

    resized_img, resized_boxes = resize(img, boxes, [640, 640])
    display_results(img, boxes, resized_img, resized_boxes)
    check_results(resized_img, resized_boxes, 'resize')
    
    # use check_results defined in utils.py for this
