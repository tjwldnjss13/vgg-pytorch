import torch


def custom_cross_entropy_loss(predict, target):
    assert predict.shape == target.shape

    predict = torch.nn.Softmax(dim=1)(predict)
    loss_tensor = -(target * torch.log2(predict + 1e-20) + (1 - target) * torch.log2(1 - predict + 1e-20))

    return loss_tensor.sum() / predict.shape[0]