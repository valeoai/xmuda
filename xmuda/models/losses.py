import numpy as np
import torch
import logging


def entropy_loss(v):
    """
        Entropy loss for probabilistic prediction vectors
        input: batch_size x classes x points
        output: batch_size x 1 x points
    """
    # (num points, num classes)
    if v.dim() == 2:
        v = v.transpose(0, 1)
        v = v.unsqueeze(0)
    # (1, num_classes, num_points)
    assert v.dim() == 3
    n, c, p = v.size()
    return -torch.sum(torch.mul(v, torch.log2(v + 1e-30))) / (n * p * np.log2(c))


def logcoral_loss(x_src, x_trg):
    """
    Geodesic loss (log coral loss), reference:
    https://github.com/pmorerio/minimal-entropy-correlation-alignment/blob/master/svhn2mnist/model.py
    :param x_src: source features of size (N, ..., F), where N is the batch size and F is the feature size
    :param x_trg: target features of size (N, ..., F), where N is the batch size and F is the feature size
    :return: geodesic distance between the x_src and x_trg
    """
    # check if the feature size is the same, so that the covariance matrices will have the same dimensions
    assert x_src.shape[-1] == x_trg.shape[-1]
    assert x_src.dim() >= 2
    batch_size = x_src.shape[0]
    if x_src.dim() > 2:
        # reshape from (N1, N2, ..., NM, F) to (N1 * N2 * ... * NM, F)
        x_src = x_src.flatten(end_dim=-2)
        x_trg = x_trg.flatten(end_dim=-2)

    # subtract the mean over the batch
    x_src = x_src - torch.mean(x_src, 0)
    x_trg = x_trg - torch.mean(x_trg, 0)

    # compute covariance
    factor = 1. / (batch_size - 1)

    cov_src = factor * torch.mm(x_src.t(), x_src)
    cov_trg = factor * torch.mm(x_trg.t(), x_trg)

    # dirty workaround to prevent GPU memory error due to MAGMA (used in SVD)
    # this implementation achieves loss of zero without creating a fork in the computation graph
    # if there is a nan or big number in the cov matrix, use where (not if!) to set cov matrix to identity matrix
    condition = (cov_src > 1e30).any() or (cov_trg > 1e30).any() or torch.isnan(cov_src).any() or torch.isnan(cov_trg).any()
    cov_src = torch.where(torch.full_like(cov_src, condition, dtype=torch.uint8), torch.eye(cov_src.shape[0], device=cov_src.device), cov_src)
    cov_trg = torch.where(torch.full_like(cov_trg, condition, dtype=torch.uint8), torch.eye(cov_trg.shape[0], device=cov_trg.device), cov_trg)

    if condition:
        logger = logging.getLogger('xmuda.train')
        logger.info('Big number > 1e30 or nan in covariance matrix, return loss of 0 to prevent error in SVD decomposition.')

    _, e_src, v_src = cov_src.svd()
    _, e_trg, v_trg = cov_trg.svd()

    # nan can occur when taking log of a value near 0 (problem occurs if the cov matrix is of low rank)
    log_cov_src = torch.mm(v_src, torch.mm(torch.diag(torch.log(e_src)), v_src.t()))
    log_cov_trg = torch.mm(v_trg, torch.mm(torch.diag(torch.log(e_trg)), v_trg.t()))

    # Frobenius norm
    return torch.mean((log_cov_src - log_cov_trg) ** 2)
