import numpy as np

from utils import get_data, check_results


def calculate_ious(gt_bboxes, pred_bboxes):
    """
    calculate ious between 2 sets of bboxes 
    args:
    - gt_bboxes [array]: Nx4 ground truth array
    - pred_bboxes [array]: Mx4 pred array
    returns:
    - iou [array]: NxM array of ious
    """
    ious = np.zeros((gt_bboxes.shape[0], pred_bboxes.shape[0]))
    for i, gt_bbox in enumerate(gt_bboxes):
        for j, pred_bbox in enumerate(pred_bboxes):
            ious[i,j] = calculate_iou(gt_bbox, pred_bbox)
    return ious

def get_all_points(box: np.ndarray):
    box_points = np.array([
        np.array([box[0], box[1]]),
        np.array([box[0], box[3]]),
        np.array([box[2], box[3]]),
        np.array([box[2], box[1]])])
    return box_points

def calculate_area(box: np.ndarray):
    width = box[2] - box[0]
    height = box[3] - box[1]
    return width * height

def calculate_iou(gt_bbox, pred_bbox):
    """
    calculate iou 
    args:
    - gt_bbox [array]: 1x4 single gt bbox
    - pred_bbox [array]: 1x4 single pred bbox
    returns:
    - iou [float]: iou between 2 bboxes
    """
    ## IMPLEMENT THIS FUNCTION
    x_diff = max(0, min(gt_bbox[2], pred_bbox[2]) - max(gt_bbox[0], pred_bbox[0]))
    y_diff = max(0, min(gt_bbox[3], pred_bbox[3]) - max(gt_bbox[1], pred_bbox[1]))
    intersection = x_diff * y_diff
    union = calculate_area(gt_bbox) + calculate_area(pred_bbox) - intersection
    return intersection/union


if __name__ == "__main__": 
    ground_truth, predictions = get_data()
    # get bboxes array
    filename = 'segment-1231623110026745648_480_000_500_000_with_camera_labels_38.png'
    gt_bboxes = [g['boxes'] for g in ground_truth if g['filename'] == filename][0]
    gt_bboxes = np.array(gt_bboxes)
    pred_bboxes = [p['boxes'] for p in predictions if p['filename'] == filename][0]
    pred_boxes = np.array(pred_bboxes)
    
    ious = calculate_ious(gt_bboxes, pred_boxes)
    check_results(ious)