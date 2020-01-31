import torch
import torch.nn.functional as F
import torchvision.transforms as T

from util.constants import *

def log_detection(logger, image, detection, target,
                  original_size, image_name,
                  images_are_loggable=False,
                  ignore_width=0, visualize_border=False):
    """
    Log images and optionally detection to tensorboard
    :param logger: [Tensorboard Logger] Tensorboard logger object.
    :param images: [tensor] batch of images indexed
                   [batch, channel, size1, size2]
    :param detections: [tensor] detected boxes
    :param targets: [tensor] ground truth boxes
    :param original_size: [int] original size of the image to be rescaled
    :param image_names: [List<str>] List of names for images.
    :param images_are_loggable: [bool] Whether images are already loggable.
    :param ignore_width: [int] Width of border where predictions/targets
                        are ignored.
    :param visualize_border: [bool] Whether to visualize ignore border.
    """
    if not images_are_loggable:
        images = prep_images_for_logging(image)

    if detection is None:
        logger.experiment.add_image(
            image_name,
            image)
    else:
        target = index2target[i]
        if len(target) == 0:
            # There was a detection but no true bbox.
            if visualize_border:
                # Plot ignore boundary
                target = [torch.Tensor([ignore_width,
                                        ignore_width,
                                        original_size - ignore_width,
                                        original_size - ignore_width])]
                target = torch.stack(target)
            else:
                target = None
        else:
            if visualize_border:
                # Plot ignore boundary
                target.append(torch.Tensor([ignore_width,
                                            ignore_width,
                                            original_size - ignore_width,
                                            original_size - ignore_width]))
            target = torch.stack(target)
        loggable_boxes, labels = prep_boxes_for_logging(
            detections=detection,
            targets=target,
            original_size=original_size)

        logger.experiment.add_image_with_boxes(
            image_name,
            image,
            loggable_boxes,
                labels=labels)

def prep_images_for_logging(image, mean=None,
                            std=None, new_size=125):
    """
    Prepare images to be logged
    :param images: [tensor] batch of images indexed
                   [channel, size1, size2]
    :param mean: [list] mean values used to normalize images
    :param std: [list] standard deviation values used to normalize images
    :param new_size: [int] new size of the image to be rescaled
    :return: images that are reversely normalized
    """
    if mean is None: mean = [0, 0, 0]
    if std is None: std = [1, 1, 1]
    image = normalize_inverse(image, mean, std)
    image_log = F.interpolate(image.unsqueeze(0), size=new_size, mode='bilinear').squeeze(0)
    return image_log

def normalize_inverse(image, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    Reverse Normalization of Pytorch Tensor
    :param images: [tensor] batch of images indexed
                   [batch, channel, size1, size2]
    :param mean: [list] mean values used to normalize images
    :param std: [list] standard deviation values used to normalize images
    :return: images that are reversely normalized
    """
    mean_inv = torch.FloatTensor(
        [-m/s for m, s in zip(mean, std)]).view(3, 1, 1)
    std_inv = torch.FloatTensor([1/s for s in std]).view(3, 1, 1)
    if torch.cuda.is_available():
        mean_inv = mean_inv.cuda()
        std_inv = std_inv.cuda()
    return (image - mean_inv) / std_inv

def prep_boxes_for_logging(detections, targets, original_size=512, new_size=125):
    """
    Resizing boxes to desired size
    :param detections: [tensor] detected boxes
    :param targets: [tensor] ground truth boxes
    :param original_size: [int] original size of the image to be rescaled
    :param new_size: [int] new size of the image to be rescaled
    :return: boxes that are rescaled
    """
    loggable_detections = detections[..., :4]*new_size/original_size
    labels = ["p"] * loggable_detections.shape[0]

    if targets is None:
        return loggable_detections, labels
    
    loggable_targets = targets*new_size/original_size
    labels += ["g"] * loggable_targets.shape[0]

    return (torch.cat([loggable_detections,
                       loggable_targets]),
            labels)

def pretty_detection(detections):
    for d in detections: 
        label, confidence, box = d
        print(f"Detect [{VOC_LABEL2NAME[label]}], [{confidence:.2f}]@[{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}]")
