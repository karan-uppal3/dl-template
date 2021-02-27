import torch
import torch.nn as nn
import numpy as np
import wandb

segmentation_classes = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
    'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
    'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'void'
]


def labels():
    l = {}
    for i, label in enumerate(segmentation_classes):
        l[i] = label
    return l


def wb_mask(bg_img, pred_mask, true_mask):
    """
    Util function for generating interactive image mask from components
    """
    return wandb.Image(bg_img, masks={
        "prediction": {"mask_data": pred_mask, "class_labels": labels()},
        "ground truth": {"mask_data": true_mask, "class_labels": labels()}})


def to_device(data, device):
    """
    Move tensor(s) to chosen device
    """
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def get_confusion_matrix(gt_label, pred_label, class_num):
    """
    Calcute the confusion matrix by given label and pred
    :param gt_label: the ground truth label
    :param pred_label: the pred label
    :param class_num: the nunber of class
    :return: the confusion matrix
    """
    index = (gt_label * class_num + pred_label).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((class_num, class_num))

    for i_label in range(class_num):
        for i_pred_label in range(class_num):
            cur_index = i_label * class_num + i_pred_label
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred_label] = label_count[cur_index]

    return confusion_matrix


def evaulate(model, data_loader, n_classes=19, criterion=nn.CrossEntropyLoss(ignore_index=255)):

    epoch_loss = 0.0
    confusion_matrix = np.zeros((n_classes, n_classes))
    k = 0
    mask_list = []

    for img, label in data_loader:

        model.eval()

        xb = to_device(img, 'cuda')
        yb = model(xb)

        loss = criterion(yb, to_device(label.long(), 'cuda'))
        epoch_loss += loss.item()

        pred = nn.functional.softmax(yb, dim=1)
        pred = torch.argmax(pred, dim=1).squeeze(1)

        preds = np.array(pred.cpu())
        label = np.array(label.cpu())

        if k < 5:
            pred_img, gt_img = preds.copy()[0], label.copy()[0]
            mask_list.append(wb_mask(img.cpu(), pred_img, gt_img))
            k += 1

        ignore_index = label != 255
        label = label[ignore_index]
        preds = preds[ignore_index]
        confusion_matrix += get_confusion_matrix(label, preds, n_classes)

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)

    IU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IU = IU_array.mean()

    #print({'meanIU':mean_IU, 'IU_array':IU_array})

    return {"Mean IOU": mean_IU, "Loss": epoch_loss/len(data_loader), "IoU": IU_array, "Predictions": mask_list}
