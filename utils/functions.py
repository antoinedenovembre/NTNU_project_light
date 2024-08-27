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
import seaborn as sns
import shutil

# Custom files
from utils.constants import *
from utils.logger import _app_logger

# ==================================================== FUNCTIONS ====================================================

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

    net = EfficientDet(config, pretrained_backbone=False)
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
    predictions, image_size=IMG_SIZE[0], iou_thr=0.44, skip_box_thr=0.43, weights=None
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

def plot_precision_recall_curve(predictions, ground_truths, iou_threshold=0.5):
    """
    Plot the Precision-Recall curve for the given predictions and ground truths.

    Args:
    - predictions: List of tuples (predicted_bboxes, predicted_class_confidences, predicted_class_labels)
    - ground_truths: List of ground truth bounding boxes for each image
    - iou_threshold: Intersection over Union (IoU) threshold to consider a detection as true positive
    """
    # Generate a color map
    colors = plt.cm.viridis(np.linspace(0, 1, len(predictions)))

    plt.figure(figsize=(8, 6))

    for i in range(len(predictions)):
        pred_boxes, pred_scores, _ = predictions[i]
        gt_boxes = ground_truths[i]

        # Convert pred_scores to a NumPy array
        pred_scores = np.array(pred_scores)

        tp = np.zeros(len(pred_boxes))
        fp = np.zeros(len(pred_boxes))

        for j, pred_box in enumerate(pred_boxes):
            best_iou = 0
            for gt_box in gt_boxes:
                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou

            if best_iou > iou_threshold:
                tp[j] = 1
            else:
                fp[j] = 1

        # Sort by scores (descending)
        indices = np.argsort(-pred_scores)
        tp = tp[indices]
        fp = fp[indices]

        # Cumulative sum
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        recalls = tp_cumsum / len(gt_boxes)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

        # Plot the Precision-Recall curve for this set of predictions
        plt.plot(recalls, precisions, marker='.', color=colors[i], label=f'Curve {i+1}')

    plt.title('Precision-Recall')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.savefig(PRECISION_RECALL_FULL_PATH)

def compute_iou(box1, box2):
        """Compute IoU between two bounding boxes."""
        x1, y1, w1, h1 = box1
        x2, y2, w2,h2 = box2

        # Determine the coordinates of the intersection rectangle
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        intersection = inter_width * inter_height

        # Union area
        box1_area = w1 * h1
        box2_area = w2 * h2
        union = box1_area + box2_area - intersection

        # IoU
        iou = intersection / union
        return iou

def plot_detection_performance_curve(predictions, ground_truths, iou_thresholds=np.linspace(0.1, 0.9, 9)):
    """
    Plot the False Negative Rate (FNR) and True Positive Rate (TPR) against IoU thresholds.

    Args:
    - predictions: List of tuples (predicted_bboxes, predicted_class_confidences, predicted_class_labels)
    - ground_truths: List of ground truth bounding boxes for each image
    - iou_thresholds: List of IoU thresholds to consider for the plot
    """

    fnr_list = []
    tpr_list = []

    for threshold in iou_thresholds:
        fn_count = 0
        tp_count = 0
        total_gt = 0

        for i in range(len(predictions)):
            pred_boxes, _, _ = predictions[i]
            gt_boxes = ground_truths[i]
            total_gt += len(gt_boxes)

            for gt_box in gt_boxes:
                best_iou = 0
                for pred_box in pred_boxes:
                    iou = compute_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou

                if best_iou >= threshold:
                    tp_count += 1
                else:
                    fn_count += 1

        fnr = fn_count / total_gt
        tpr = tp_count / total_gt
        fnr_list.append(fnr)
        tpr_list.append(tpr)

    plt.figure(figsize=(10, 6))

    plt.plot(iou_thresholds, fnr_list, marker='o', label='False Negative Rate (FNR)', color='red')
    plt.plot(iou_thresholds, tpr_list, marker='o', label='True Positive Rate (TPR)', color='green')

    plt.title('Detection Performance')
    plt.xlabel('IoU Threshold')
    plt.ylabel('Rate')
    plt.legend()
    plt.grid(True)
    plt.savefig(DETECTION_PERF_FULL_PATH)

def plot_f1_score_curve(predictions, ground_truths, iou_thresholds=np.linspace(0.1, 0.9, 9)):
    """
    Plot the F1 Score against IoU thresholds.

    Args:
    - predictions: List of tuples (predicted_bboxes, predicted_class_confidences, predicted_class_labels)
    - ground_truths: List of ground truth bounding boxes for each image
    - iou_thresholds: List of IoU thresholds to consider for the plot
    """
    f1_scores = []

    for threshold in iou_thresholds:
        tp_count = 0
        fp_count = 0
        fn_count = 0

        for i in range(len(predictions)):
            pred_boxes, _, _ = predictions[i]
            gt_boxes = ground_truths[i]

            matched_gt = np.zeros(len(gt_boxes))

            for pred_box in pred_boxes:
                best_iou = 0
                best_gt_idx = -1
                for j, gt_box in enumerate(gt_boxes):
                    iou = compute_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j

                if best_iou >= threshold:
                    if matched_gt[best_gt_idx] == 0:  # If this GT hasn't been matched yet
                        tp_count += 1
                        matched_gt[best_gt_idx] = 1  # Mark this GT as matched
                    else:
                        fp_count += 1
                else:
                    fp_count += 1

            fn_count += np.sum(matched_gt == 0)

        # Calculate precision, recall, and F1 score
        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1_score)

    plt.figure(figsize=(10, 6))

    plt.plot(iou_thresholds, f1_scores, marker='o', label='F1 Score', color='blue')

    plt.title('F1 Score against IoU Threshold')
    plt.xlabel('IoU Threshold')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(F1_SCORE_FULL_PATH)

def plot_roc_curve(predictions, ground_truths, iou_thresholds=np.linspace(0.1, 0.9, 9)):
    """
    Plot the FPR vs TPR curve (ROC Curve) against IoU thresholds.

    Args:
    - predictions: List of tuples (predicted_bboxes, predicted_class_confidences, predicted_class_labels)
    - ground_truths: List of ground truth bounding boxes for each image
    - iou_thresholds: List of IoU thresholds to consider for the plot
    """

    tpr_list = []
    fpr_list = []

    total_ground_truths = sum([len(gt_boxes) for gt_boxes in ground_truths])
    total_predictions = sum([len(pred_boxes) for pred_boxes, _, _ in predictions])

    for threshold in iou_thresholds:
        tp_count = 0
        fp_count = 0
        total_negatives = total_predictions - total_ground_truths  # An estimate of the negatives

        for i in range(len(predictions)):
            pred_boxes, _, _ = predictions[i]
            gt_boxes = ground_truths[i]

            matched_gt = np.zeros(len(gt_boxes))

            for pred_box in pred_boxes:
                best_iou = 0
                best_gt_idx = -1
                for j, gt_box in enumerate(gt_boxes):
                    iou = compute_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j

                if best_iou >= threshold:
                    if matched_gt[best_gt_idx] == 0:  # If this GT hasn't been matched yet
                        tp_count += 1
                        matched_gt[best_gt_idx] = 1  # Mark this GT as matched
                    else:
                        fp_count += 1
                else:
                    fp_count += 1

        fn_count = total_ground_truths - tp_count
        fpr = fp_count / total_negatives if total_negatives > 0 else 0
        tpr = tp_count / total_ground_truths if total_ground_truths > 0 else 0
        fpr_list.append(fpr)
        tpr_list.append(tpr)

    plt.figure(figsize=(10, 6))

    plt.plot(fpr_list, tpr_list, marker='o', label='FPR against TPR (ROC Curve)', color='red')

    plt.title('FPR vs TPR Curve (ROC Curve)')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.legend()
    plt.grid(True)
    plt.savefig(ROC_CURVE_FULL_PATH)

def plot_confusion_matrix(tp, fp, fn):
    """
    Plot the confusion matrix for object detection.

    Args:
    - tp: Number of true positives
    - fp: Number of false positives
    - fn: Number of false negatives
    """
    # Confusion matrix components: [TP, FN], [FP, TN (we assume TN is 0 since we focus on object detection)]
    matrix = np.array([[tp, fn], [fp, 0]])

    # Plotting the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted Positive', 'Predicted Negative'],
                yticklabels=['Actual Positive', 'Actual Negative'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig(CONFUSION_MATRIX_FULL_PATH)

def compute_confusion_matrix(predictions, ground_truths, iou_threshold=0.5):
    """
    Compute TP, FP, and FN based on IoU threshold.

    Args:
    - predictions: List of tuples (predicted_bboxes, predicted_class_confidences, predicted_class_labels)
    - ground_truths: List of ground truth bounding boxes for each image
    - iou_threshold: IoU threshold to consider for matching predictions to ground truths
    """
    tp_count = 0
    fp_count = 0
    fn_count = 0

    for i in range(len(predictions)):
        pred_boxes, _, _ = predictions[i]
        gt_boxes = ground_truths[i]

        matched_gt = set()

        for pred_box in pred_boxes:
            best_iou = 0
            best_gt_idx = -1
            for j, gt_box in enumerate(gt_boxes):
                if j in matched_gt:
                    continue
                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

            if best_iou >= iou_threshold:
                tp_count += 1
                matched_gt.add(best_gt_idx)
            else:
                fp_count += 1

        # Any ground truth that was not matched counts as a false negative
        fn_count += len(gt_boxes) - len(matched_gt)

    return tp_count, fp_count, fn_count

def get_all_metrics(predictions, ground_truths, iou_thresholds=np.linspace(0.1, 0.9, 9)):
    """
    Compute all metrics for object detection.

    Args:
    - predictions: List of tuples (predicted_bboxes, predicted_class_confidences, predicted_class_labels)
    - ground_truths: List of ground truth bounding boxes for each image
    - iou_thresholds: List of IoU thresholds to consider for the plot
    """

    # Compute : Precision, Recall, F1 Score, ROC Curve, Detection Performance Curve, Precision-Recall Curve
    plot_precision_recall_curve(predictions, ground_truths)
    plot_detection_performance_curve(predictions, ground_truths, iou_thresholds)
    plot_f1_score_curve(predictions, ground_truths, iou_thresholds)
    plot_roc_curve(predictions, ground_truths, iou_thresholds)

    # Compute confusion matrix
    tp, fp, fn = compute_confusion_matrix(predictions, ground_truths)
    plot_confusion_matrix(tp, fp, fn)

def backup_model():
    if MODEL_FULL_PATH.exists():
        shutil.copy(MODEL_FULL_PATH, MODEL_BACKUP_FULL_PATH)
        _app_logger.info(f"Model backed up to {MODEL_BACKUP_FULL_PATH}")