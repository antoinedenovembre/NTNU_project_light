# Metrics used to train and evaluate the model

## I. Introduction

This document is a summary of the metrics used to evaluate the object detection model. 
The model is evaluated on the test set using the following metrics: 
- Precision,
- Recall,
- F1 score,
- True Positive Rate (TPR),
- False Positive Rate (FPR),
- Average Precision (AP),
- Mean Average Precision (mAP).

## II. Loss function

The loss function is from the `effdet` library, which is a combination of the focal loss and the smooth L1 loss. The focal loss is used to handle the class imbalance problem, while the smooth L1 loss is used to handle the bounding box regression problem.

$$
\begin{equation}
Loss = \alpha \times \text{Focal Loss} + \beta \times \text{Smooth L1 Loss}
\end{equation}
$$

where $\alpha$ and $\beta$ are the weights of the focal loss and the smooth L1 loss, respectively.

### II.a. Definition of the Focal Loss

The focal loss is a modification of the cross-entropy loss, which is used to handle the class imbalance problem. It is defined as:

$$
\begin{equation}
\text{Focal Loss} = -\alpha \times (1 - p_t)^\gamma \times \log(p_t)
\end{equation}
$$

where $p_t$ is the predicted probability of the true class, $\alpha$ is the balancing parameter, and $\gamma$ is the focusing parameter.

### II.b. Definition of the Smooth L1 Loss

The smooth L1 loss is used to handle the bounding box regression problem. It is defined as:

$$
\begin{equation}
\text{Smooth L1 Loss} =
\begin{cases} 
0.5 \times (\text{error})^2 / \delta, & \text{if } |\text{error}| < \delta \\
|\text{error}| - 0.5 \times \delta, & \text{otherwise}
\end{cases}
\end{equation}
$$

where $\text{error}$ is the difference between the predicted and the ground truth bounding box, and $\delta$ is a hyperparameter.

## III. Evaluation metrics

### III.a. Definition of IoU

The Intersection over Union (IoU) is a metric used to evaluate the accuracy of an object detection model. It is defined as the ratio of the area of the intersection of the predicted bounding box and the ground truth bounding box to the area of their union.

$$
\begin{equation}
IoU = \frac{Area_{Intersection}}{Area_{Union}}
\end{equation}
$$

We use IoU here, giving a threshold above which the predicted bounding box is considered as a true positive (TP). If the IoU is below the threshold, the predicted bounding box is considered as a false positive (FP).

### III.b. Formulas

To evaluate the model, we use the following metrics, computed over different IoU thresholds:

#### III.b.1. Precision, Recall, F1 score

$$
\begin{equation}
Precision = \frac{TP}{TP + FP}
\end{equation}
$$

$$
\begin{equation}
Recall = \frac{TP}{TP + FN}
\end{equation}
$$

$$
\begin{equation}
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
\end{equation}
$$

#### III.b.2. True Positive Rate (TPR), False Positive Rate (FPR)

$$
\begin{equation}
TPR = \frac{TP}{TP + FN}
\end{equation}
$$

$$
\begin{equation}
FPR = \frac{FP}{FP + TN}
\end{equation}
$$

#### III.b.3. Average Precision (AP), Mean Average Precision (mAP)

$$
\begin{equation}
AP = \sum_{n} (Recall_n - Recall_{n-1}) \times Precision_n
\end{equation}
$$

$$
\begin{equation}
mAP = \frac{1}{N} \sum_{i=1}^{N} AP_i
\end{equation}
$$
