import torch
import torch.nn as nn
from utils.metrics import RunningScore
import numpy as np

def to_device(data, device):
    """
    Move tensor(s) to chosen device
    """
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def evaulate(model, data_loader, n_classes=19, criterion=nn.CrossEntropyLoss(ignore_index=255)):

    score = RunningScore(n_classes=n_classes)
    epoch_loss = 0.0

    for img, label in data_loader:

        model.eval()

        xb = to_device(img, 'cuda')
        yb = model(xb)

        loss = criterion(yb, to_device(label, 'cuda'))
        epoch_loss += loss.item()

        pred = nn.functional.softmax(yb, dim=1)
        pred = torch.argmax(pred, dim=1).squeeze(1)

        preds = np.array(pred.cpu())
        label = np.array(label.cpu())

        score.update(label, preds)

    metrics = score.get_scores()

    print("mIOU = ", metrics[0]['Mean_IoU'],"Accuracy =", metrics[0]['Overall_Acc'])

    return {"Mean IOU": metrics[0]['Mean_IoU'], "Accuracy": metrics[0]['Overall_Acc'], "Loss": epoch_loss/len(data_loader)}
