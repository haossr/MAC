import numpy as np
import torch


def nms(boxes, probs, threshold=0.01, iou_threshold=0.5):
    """
    Perform Non-maximum suppression over boxes given the probs
    - boxes: [num_boxes, [xmin, ymin, xmax, ymax]]
    - probs: [num_boxes, num_class]

    Return:
    - detections: List of detections

    """
    if boxes.size == 0:
        return []
    num_class = probs.shape[1]
    scores = np.max(probs, axis=1, keepdims=False)
    preds = np.argmax(probs, axis=1)
    selected = (scores > threshold)

    boxes = boxes[selected]
    scores = scores[selected]
    preds = preds[selected]

    detections = []
    for _label in range(num_class):
        _scores = scores[preds == _label]
        _boxes = boxes[preds == _label]

        nms_selected = torch.ops.torchvision.nms(
            torch.from_numpy(_boxes),
            torch.from_numpy(_scores),
            iou_threshold)
        for i in nms_selected:
            detections.append((_label, _scores[i], _boxes[i]))
    return detections


def nms_batch(boxes, probs, threshold=0.01, iou_threshold=0.5):
    """
    Perform Non-maximum suppression over boxes given the probs
    - boxes: [batch_size, num_boxes, [xmin, ymin, xmax, ymax]]
    - probs: [batch_size, num_boxes, num_class]

    Return:
    - detections: List of detections

    """
    B = boxes.shape[0]
    detections = [nms_tensor(b, p, threshold, iou_threshold) for b, p in zip(boxes, probs)]


def nms_tensor(boxes, probs, threshold=0.01, iou_threshold=0.5):
    """
    Perform Non-maximum suppression over boxes given the probs
    - boxes: [num_boxes, [xmin, ymin, xmax, ymax]]
    - probs: [num_boxes, num_class]

    Return:
    - detections: List of detections

    """
    if boxes.size == 0:
        return []

    C = probs.shape[1]

    scores, preds = torch.max(probs, axis=1, keepdims=False)
    selected = (scores > threshold)

    boxes = boxes[selected]
    scores = scores[selected]
    preds = preds[selected]

    detections = []
    for _label in range(num_class):
        _scores = scores[preds == _label]
        _boxes = boxes[preds == _label]

        nms_selected = torch.ops.torchvision.nms(
            _boxes,
            _scores,
            iou_threshold)
        for i in nms_selected:
            detections.append((_label, _scores[i], _boxes[i]))
    return detections
