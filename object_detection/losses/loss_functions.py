

from distutils.version import LooseVersion
import torch
import torch.nn.functional as F
from torch.autograd import Variable


class CrossEntropy2D(torch.nn.Module):
    """
        Cross Entropy loss implementaion used for training
    """

    def __init__(self):
        super(CrossEntropy2D, self).__init__()

    def forward(self, input, target, weight=None, size_average=True):
        # input: (n, c, h, w), target: (n, h, w)
        n, c, h, w = input.size()
        # log_p: (n, c, h, w)
        if LooseVersion(torch.__version__) < LooseVersion('0.3'):
            # ==0.2.X
            log_p = F.log_softmax(input)
        else:
            # >=0.3
            log_p = F.log_softmax(input, dim=1)
        # log_p: (n*h*w, c)
        log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
        log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
        log_p = log_p.view(-1, c)
        # target: (n*h*w,)
        mask = target >= 0
        target = target[mask]
        loss = F.nll_loss(log_p, target, weight=weight, reduction='sum')
        if size_average:
            loss /= mask.data.sum()
        return loss


class DiceLoss(torch.nn.Module):
    """
        Cross Entropy loss implementaion used for training
    """

    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target, weight=None, size_average=True):

        """This definition generalize to real valued pred and target vector.
        This should be differentiable.
            pred: tensor with first dimension as batch
            target: tensor with first dimension as batch
            """

        smooth = 1.
        # have to use contiguous since they may from a torch.view op
        #probs = Variable(F.softmax(pred, dim=1).data.max(1)[1].float(), requires_grad=True)
        probs = F.sigmoid(pred)
        iflat = probs.contiguous().view(-1)
        tflat = target.float().contiguous().view(-1)
        intersection = (iflat * tflat).sum()

        A_sum = torch.sum(iflat)
        B_sum = torch.sum(tflat)

        return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))

