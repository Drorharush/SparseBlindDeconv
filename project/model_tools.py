import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_tensor_0to1(t: torch.Tensor) -> torch.Tensor:
    """
    Normalizes a tensor such that the data in each sample in the batch will distribute between 0 and 1.
    """
    shape = t.size()
    t = t.view(t.size(0), -1)
    t -= t.min(1, keepdim=True)[0]
    t /= t.max(1, keepdim=True)[0]
    return t.view(shape)


def normalize_tensor_sumto1(t: torch.Tensor) -> torch.Tensor:
    """
    Normalizes a tensor such that the data in each sample in the batch will sum to 1.
    """
    norm = t.sum(dim=1, keepdim=True)
    norm[norm == 0] = 1
    return t / norm


def smooth_abs(x, eps=1e-18) -> float:
    """
    A continuesly differentiable approximation for absolute value function.
    """
    return (x ** 2 + eps) ** 0.5 - (eps ** 0.5)


class ActivationLoss(nn.Module):
    """
    loss = 0.5|pred (*) kernel - true (*) kernel|**2 + r * regulator(pred)
    """
    def __init__(self, r=0.1):
        super().__init__()
        self.r = r
        self.mu = 10 ** -6

    def regulator(self, activation):
        return torch.sum(self.mu ** 2 * (torch.sqrt(1 + (self.mu ** -2) * torch.abs(activation)) - 1))

    def forward(self, activation_pred: torch.Tensor, activation: torch.Tensor, kernel: torch.Tensor):
        conv_pred = []
        conv_target = []
        for i in range(activation_pred.shape[0]):
            single_kernel = kernel[i].unsqueeze(dim=0)
            single_activation_pred = activation_pred[i].unsqueeze(dim=0)
            single_activation = activation[i].unsqueeze(dim=0).unsqueeze(dim=0)
            # plot_conv(single_kernel, single_activation, target[i])
            conv_pred.append(F.conv2d(single_activation_pred, single_kernel, padding='same'))
            conv_target.append(F.conv2d(single_activation, single_kernel, padding='same'))
        conv_pred_stack = torch.stack(conv_pred, dim=0).squeeze(dim=1)
        conv_target_stack = torch.stack(conv_target, dim=0).squeeze(dim=1)
        regulation_term = smooth_abs(self.regulator(activation_pred) - self.regulator(activation))

        loss = F.huber_loss(conv_pred_stack, conv_target_stack) / 2
        loss += self.r * regulation_term
        loss /= activation.shape[0]
        if torch.cuda.is_available():
            return loss.cuda()
        return loss
