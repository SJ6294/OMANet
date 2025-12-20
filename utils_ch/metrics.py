
from sklearn.metrics import confusion_matrix
import torch

def calculate_metrics(gt, pred): 
    """
    input : tensor
    gt: shape (B, C, H, W) 또는 (1, C, H, W)  # train, valid
    pred: shape (B, C, H, W) 또는 (1, C, H, W)
    """
    pred = (torch.sigmoid(pred) > 0.5).float()
    gt = ((gt) > 0.5).float()
            
    if gt.dim() == 4:  # (B, C, H, W)
        gt = gt.squeeze(1)   # (B, H, W)
        pred = pred.squeeze(1)
    
    gt = gt.cpu().numpy()
    pred = pred.cpu().numpy()
    
    # flatten
    gt_flat = gt.flatten()
    pred_flat = pred.flatten()
    
    conf_matrix = confusion_matrix(gt_flat, pred_flat, labels=[0, 1])
    tn = conf_matrix[0, 0]
    fp = conf_matrix[0, 1]
    fn = conf_matrix[1, 0]
    tp = conf_matrix[1, 1]
    
    background_iou = tn / (tn + fp + fn + 1e-10)
    object_iou = tp / (tp + fp + fn + 1e-10)
    miou = (background_iou + object_iou) / 2
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)
    

    results = {
        "background_iou": background_iou,
        "object_iou": object_iou,
        "miou": miou,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }

    return results
