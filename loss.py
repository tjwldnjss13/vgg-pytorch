import torch


def custom_cross_entropy_loss(predict, target):
    assert predict.shape == target.shape

    n_batch = predict.shape[0]
    losses = -(target * torch.log2(predict + 1e-20) + (1 - target) * torch.log2(1 - predict + 1e-20))

    return losses.sum() / n_batch