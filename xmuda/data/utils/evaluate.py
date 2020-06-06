import numpy as np
from sklearn.metrics import confusion_matrix as CM

class Evaluator(object):
    def __init__(self, class_names, labels=None):
        self.class_names = tuple(class_names)
        self.num_classes = len(class_names)
        self.labels = np.arange(self.num_classes) if labels is None else np.array(labels)
        assert self.labels.shape[0] == self.num_classes
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def update(self, pred_label, gt_label):
        """Update per instance

        Args:
            pred_label (np.ndarray): (num_points)
            gt_label (np.ndarray): (num_points,)

        """
        # convert ignore_label to num_classes
        # refer to sklearn.metrics.confusion_matrix
        gt_label[gt_label == -100] = self.num_classes
        confusion_matrix = CM(gt_label.flatten(),
                              pred_label.flatten(),
                              labels=self.labels)
        self.confusion_matrix += confusion_matrix

    def batch_update(self, pred_labels, gt_labels):
        assert len(pred_labels) == len(gt_labels)
        for pred_label, gt_label in zip(pred_labels, gt_labels):
            self.update(pred_label, gt_label)

    @property
    def overall_acc(self):
        return np.sum(np.diag(self.confusion_matrix)) / np.sum(self.confusion_matrix)

    @property
    def overall_iou(self):
        class_iou = np.array(self.class_iou.copy())
        class_iou[np.isnan(class_iou)] = 0
        return np.mean(class_iou)

    @property
    def class_seg_acc(self):
        return [self.confusion_matrix[i, i] / np.sum(self.confusion_matrix[i])
                for i in range(self.num_classes)]

    @property
    def class_iou(self):
        iou_list = []
        for i in range(self.num_classes):
            tp = self.confusion_matrix[i, i]
            p = self.confusion_matrix[:, i].sum()
            g = self.confusion_matrix[i, :].sum()
            union = p + g - tp
            if union == 0:
                iou = float('nan')
            else:
                iou = tp / union
            iou_list.append(iou)
        return iou_list

    def print_table(self):
        from tabulate import tabulate
        header = ['Class', 'Accuracy', 'IOU', 'Total']
        seg_acc_per_class = self.class_seg_acc
        iou_per_class = self.class_iou
        table = []
        for ind, class_name in enumerate(self.class_names):
            table.append([class_name,
                          seg_acc_per_class[ind] * 100,
                          iou_per_class[ind] * 100,
                          int(self.confusion_matrix[ind].sum()),
                          ])
        return tabulate(table, headers=header, tablefmt='psql', floatfmt='.2f')

    def save_table(self, filename):
        from tabulate import tabulate
        header = ('overall acc', 'overall iou') + self.class_names
        table = [[self.overall_acc, self.overall_iou] + self.class_iou]
        with open(filename, 'w') as f:
            # In order to unify format, remove all the alignments.
            f.write(tabulate(table, headers=header, tablefmt='tsv', floatfmt='.5f',
                             numalign=None, stralign=None))
