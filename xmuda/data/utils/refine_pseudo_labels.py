import torch


def refine_pseudo_labels(probs, pseudo_label, ignore_label=-100):
    """
    Reference: https://github.com/liyunsheng13/BDL/blob/master/SSL.py
    Per class, set the less confident half of labels to ignore label.
    :param probs: maximum probabilities (N,), where N is the number of 3D points
    :param pseudo_label: predicted label which had maximum probability (N,)
    :param ignore_label:
    :return:
    """
    probs, pseudo_label = torch.tensor(probs), torch.tensor(pseudo_label)
    for cls_idx in pseudo_label.unique():
        curr_idx = pseudo_label == cls_idx
        curr_idx = curr_idx.nonzero().squeeze(1)
        thresh = probs[curr_idx].median()
        thresh = min(thresh, 0.9)
        ignore_idx = curr_idx[probs[curr_idx] < thresh]
        pseudo_label[ignore_idx] = ignore_label
    return pseudo_label.numpy()
