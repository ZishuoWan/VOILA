import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
from skimage.measure import label

class VoxelF1Loss(nn.Module):
    def __init__(self, apply_nonlin, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True, clip_tp: float = None):
        """
        """
        super(VoxelF1Loss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.clip_tp = clip_tp
        self.ddp = ddp

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin:
            x = torch.softmax(x, 1)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        if self.ddp and self.batch_dice:
            tp = AllGatherGrad.apply(tp).sum(0)
            fp = AllGatherGrad.apply(fp).sum(0)
            fn = AllGatherGrad.apply(fn).sum(0)

        if self.clip_tp is not None:
            tp = torch.clip(tp, min=self.clip_tp , max=None)

        nominator = 2 * tp
        denominator = 2 * tp + fp + fn

        dc = (nominator + self.smooth) / (torch.clip(denominator + self.smooth, 1e-8))

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return 1 - dc

class CEforPixelContrast(nn.Module):
    def __init__(self, weight=None, ignore_index=None, num_classes=3, **kwargs):
        super(CEforPixelContrast, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, **self.kwargs)
        self.criterion = nn.BCEWithLogitsLoss()

    def _dice(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        dice = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        return dice
    
    def forward(self, predict, target, label_idx=None, mask=None):
        if isinstance(predict, (list, tuple)):
            if self.weight is None:
                weight = torch.tensor([1 / (2 ** i) for i in range(len(predict))],dtype=torch.float32)
                weight[-1] = 0
                self.weight = weight / weight.sum()
            for i in range(len(predict)):
                if i == 0:
                    total_loss = self.weight[i] * self.forward_percase(predict[i], target[i], label_idx=label_idx)
                    continue
                total_loss += self.weight[i] * self.forward_percase(predict[i], target[i], label_idx=label_idx)
            return total_loss, torch.zeros_like(total_loss)
        else:
            l = self.forward_percase(predict, target, label_idx=label_idx, mask=mask)
            return l, torch.zeros_like(l)

    def forward_percase(self, predict, target, label_idx=None, mask=None):
        if len(predict.shape) == len(target.shape): target = target.float()
        else: target = target.long()
        if mask is not None:
            target[mask==False] = self.ignore_index
        mean_loss = self.ce(predict, target)

        return mean_loss

def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    return dice, hd95


def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if net_output.shape == gt.shape:
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x, device=net_output.device)
            y_onehot.scatter_(1, gt, 1)

    if len(axes) > 0:
        if mask is not None:
            with torch.no_grad():
                mask_here = torch.tile(mask, (1, net_output.shape[1], *[1 for i in range(2, len(net_output.shape))]))
            tp = torch.sum((net_output * y_onehot * mask_here), dim=axes, keepdim=False)
            fp = torch.sum((net_output * (1 - y_onehot) * mask_here), dim=axes, keepdim=False)
            fn = torch.sum(((1 - net_output) * y_onehot * mask_here), dim=axes, keepdim=False)
            tn = torch.sum(((1 - net_output) * (1 - y_onehot) * mask_here), dim=axes, keepdim=False)
        else:
            tp = torch.sum((net_output * y_onehot), dim=axes, keepdim=False)
            fp = torch.sum((net_output * (1 - y_onehot)), dim=axes, keepdim=False)
            fn = torch.sum(((1 - net_output) * y_onehot), dim=axes, keepdim=False)
            tn = torch.sum(((1 - net_output) * (1 - y_onehot)), dim=axes, keepdim=False)

    return tp, fp, fn, tn

class AllGatherGrad(torch.autograd.Function):
    # stolen from pytorch lightning
    @staticmethod
    def forward(
        ctx,
        tensor: torch.Tensor,
        group = None,
    ) -> torch.Tensor:
        ctx.group = group

        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]

        torch.distributed.all_gather(gathered_tensor, tensor, group=group)
        gathered_tensor = torch.stack(gathered_tensor, dim=0)

        return gathered_tensor

    @staticmethod
    def backward(ctx, *grad_output: torch.Tensor):
        grad_output = torch.cat(grad_output)

        torch.distributed.all_reduce(grad_output, op=torch.distributed.ReduceOp.SUM, async_op=False, group=ctx.group)

        return grad_output[torch.distributed.get_rank()], None
