# ==================================================== IMPORTS ====================================================

# Libraries
from pathlib import Path
from matplotlib import patches
from pathlib import Path
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet
from effdet.soft_nms import soft_nms
from albumentations.pytorch.transforms import ToTensorV2
from ensemble_boxes import ensemble_boxes_wbf
from effdet.efficientdet import HeadNet
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import albumentations as A

# Custom files
from utils.constants import *
from utils.logger import _app_logger

# ==================================================== FUNCTIONS ====================================================

img_size = (512, 512)

def get_rectangle_edges_from_pascal_bbox(bbox):
    xmin_top_left, ymin_top_left, xmax_bottom_right, ymax_bottom_right = bbox

    bottom_left = (xmin_top_left, ymax_bottom_right)
    width = xmax_bottom_right - xmin_top_left
    height = ymin_top_left - ymax_bottom_right

    return bottom_left, width, height

def get_pascal_bbox(bbox):
    xmin, ymin = bbox[0], bbox[1]
    width, height = bbox[2], bbox[3]
    xmax = xmin + width
    ymax = ymin + height
    return [xmin, ymin, xmax, ymax]

def get_pascal_bboxes(bboxes):
    # coco -> pascal
    out = []
    for bbox in bboxes:
        out.append(get_pascal_bbox(bbox))
    return out

def draw_pascal_voc_bboxes(
    plot_ax,
    bboxes,
    confidences=None,
    get_rectangle_corners_fn=get_rectangle_edges_from_pascal_bbox,
):
    for i, bbox in enumerate(bboxes):
        bottom_left, width, height = get_rectangle_corners_fn(bbox)

        rect_1 = patches.Rectangle(
            bottom_left,
            width,
            height,
            linewidth=4,
            edgecolor="black",
            fill=False,
        )
        rect_2 = patches.Rectangle(
            bottom_left,
            width,
            height,
            linewidth=2,
            edgecolor="white",
            fill=False,
        )

        if confidences:
            rx, ry = rect_2.get_xy()
            plot_ax.annotate(
                np.round(confidences[i], 2), (rx, ry), color="red", weight="bold"
            )

        # Add the patch to the Axes
        plot_ax.add_patch(rect_1)
        plot_ax.add_patch(rect_2)

def show_image(
    image, bboxes=None, draw_bboxes_fn=draw_pascal_voc_bboxes, figsize=(10, 10)
):
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image)

    if bboxes is not None:
        draw_bboxes_fn(ax, bboxes)

    plt.show()

def create_model(num_classes=1, image_size=512, architecture="tf_efficientnetv2_l"):
    # efficientdet_model_param_dict["tf_efficientnetv2_l"] = dict(
    #     name="tf_efficientnetv2_l",
    #     backbone_name="tf_efficientnetv2_l",
    #     backbone_args=dict(drop_path_rate=0.2),
    #     num_classes=num_classes,
    #     url="",
    # )

    config = get_efficientdet_config(architecture)
    config.update({"num_classes": num_classes})
    config.update({"image_size": (image_size, image_size)})

    _app_logger.debug(config)

    net = EfficientDet(config, pretrained_backbone=True)
    net.class_net = nn.Identity()
    net.box_net = nn.Identity()
    # net.load_state_dict(
    #     torch.load("results/weights/barlow_pretrained_weights_backbone")
    # )
    net.class_net = HeadNet(config, num_outputs=config.num_classes)
    net.box_net = HeadNet(config, num_outputs=4)
    return DetBenchTrain(net, config)

def get_train_transforms(target_img_size=512):
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Resize(height=target_img_size, width=target_img_size, p=1),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2(p=1),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
        ),
    )

def get_valid_transforms(target_img_size=512):
    return A.Compose(
        [
            A.Resize(height=target_img_size, width=target_img_size, p=1),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2(p=1),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
        ),
    )

def run_wbf(
    predictions, image_size=img_size[0], iou_thr=0.44, skip_box_thr=0.43, weights=None
):
    bboxes = []
    confidences = []
    class_labels = []

    for prediction in predictions:
        boxes = [(prediction["boxes"] / image_size).tolist()]
        scores = [prediction["scores"].tolist()]
        labels = [prediction["classes"].tolist()]

        boxes, scores, labels = ensemble_boxes_wbf.weighted_boxes_fusion(
            boxes,
            scores,
            labels,
            weights=weights,
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr,
        )
        boxes = boxes * (image_size - 1)
        bboxes.append(boxes.tolist())
        confidences.append(scores.tolist())
        class_labels.append(labels.tolist())

    return bboxes, confidences, class_labels

def run_soft_nms(predictions, iou_thr=0.44, skip_box_thr=0.12, sigma=0.5):
    bboxes = []
    confidences = []
    class_labels = []

    for prediction in predictions:
        boxes = prediction["boxes"]
        scores = prediction["scores"]
        labels = prediction["classes"]

        idxs_out, scores_out = soft_nms(
            boxes,
            scores,
            iou_threshold=iou_thr,
            score_threshold=skip_box_thr,
            sigma=sigma,
        )
        bboxes.append(boxes[idxs_out].tolist())
        confidences.append(scores_out.tolist())
        class_labels.append(labels[idxs_out].tolist())

    return bboxes, confidences, class_labels

def compare_bboxes_for_image(
    image,
    predicted_bboxes,
    predicted_class_confidences,
    actual_bboxes,
    image_id,
    draw_bboxes_fn=draw_pascal_voc_bboxes,
    figsize=(20, 20),
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ax1.imshow(image)
    ax1.set_title("Actual")
    ax2.imshow(image)
    ax2.set_title("Prediction")

    draw_bboxes_fn(ax1, [], None)
    draw_bboxes_fn(ax2, predicted_bboxes, predicted_class_confidences)

    # Ensure the results directory exists
    results_path_img = IMG_DIR
    results_path_img.mkdir(parents=True, exist_ok=True)

    # Save the image with a unique filename
    output_filename = results_path_img / f"result_{image_id}.png"
    plt.savefig(output_filename)
    plt.close(fig)  # Close the figure to free memory

    plt.show()
