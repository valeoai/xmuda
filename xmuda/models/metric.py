import torch
from xmuda.common.utils.metric_logger import AverageMeter


class SegAccuracy(AverageMeter):
    """Segmentation accuracy"""
    name = 'seg_acc'
    
    def __init__(self, ignore_index=-100):
        super(SegAccuracy, self).__init__()
        self.ignore_index = ignore_index

    def update_dict(self, preds, labels):
        seg_logit = preds['seg_logit']  # (b, c, n)
        seg_label = labels['seg_label']  # (b, n)
        pred_label = seg_logit.argmax(1)

        mask = (seg_label != self.ignore_index)
        seg_label = seg_label[mask]
        pred_label = pred_label[mask]

        tp_mask = pred_label.eq(seg_label)  # (b, n)
        self.update(tp_mask.sum().item(), tp_mask.numel())


class SegIoU(object):
    """Segmentation IoU
    References: https://github.com/pytorch/vision/blob/master/references/segmentation/utils.py
    """

    def __init__(self, num_classes, ignore_index=-100, name='seg_iou'):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.mat = None
        self.name = name

    def update_dict(self, preds, labels):
        seg_logit = preds['seg_logit']  # (batch_size, num_classes, num_points)
        seg_label = labels['seg_label']  # (batch_size, num_points)
        pred_label = seg_logit.argmax(1)

        mask = (seg_label != self.ignore_index)
        seg_label = seg_label[mask]
        pred_label = pred_label[mask]

        # Update confusion matrix
        # TODO: Compare the speed between torch.histogram and torch.bincount after pytorch v1.1.0
        n = self.num_classes
        with torch.no_grad():
            if self.mat is None:
                self.mat = seg_label.new_zeros((n, n))
            inds = n * seg_label + pred_label
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def reset(self):
        self.mat = None

    @property
    def iou(self):
        h = self.mat.float()
        iou = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return iou

    @property
    def global_avg(self):
        return self.iou.mean().item()

    @property
    def avg(self):
        return self.global_avg

    def __str__(self):
        return '{iou:.4f}'.format(iou=self.iou.mean().item())

    @property
    def summary_str(self):
        return str(self)
