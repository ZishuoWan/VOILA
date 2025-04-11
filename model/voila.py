import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from functools import partial
import numpy as np
from model.cvae import ConditionalVAE
from scipy.ndimage import gaussian_filter
import einops

def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class VOILA(nn.Module):

    def __init__(self,
                 block = Bottleneck,
                 layers = [3, 4, 23, 3],
                 stem_features = 16,
                 num_classes = 118,
                 fpn_dim = 32,
                 shortcut_type='B',
                 text_features = None,
                 image_shape=128,
                 use_cas = False,
                 **args):
        self.stem_features = stem_features
        self.use_cas = use_cas
        self.image_shape = image_shape
        self.do_vli = False
        super(VOILA, self).__init__()
        self.conv1 = nn.Conv3d(
            1,
            stem_features,
            kernel_size=7,
            stride=2,
            padding=(3, 3, 3),
            bias=False)
            
        self.bn1 = nn.BatchNorm3d(stem_features)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, stem_features, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, stem_features*2, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, stem_features*4, layers[2], shortcut_type, stride=2, dilation=2)
        self.layer4 = self._make_layer(
            block, stem_features*8, layers[3], shortcut_type, stride=2, dilation=4)

        input_shape = {}
        for l in range(4):
            scale_factor = 2**(l+1)
            input_shape[f'p{l+2}'] = (self.stem_features//(2**(3-l)), tuple([image_shape//scale_factor]*3), scale_factor)


        self.fpn = SemSegFPNHead(
            input_shape, num_classes=fpn_dim, conv_dims=fpn_dim, common_stride=1, norm=nn.BatchNorm3d, conv_op=nn.Conv3d
        )
        if text_features is not None:
            self.do_vli = True
            self.text_features = text_features
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            self.text_projection = nn.Parameter(torch.empty(512, fpn_dim))
            nn.init.normal_(self.text_projection, std=512 ** -0.5)
        else:
            self.seg_out = nn.Linear(fpn_dim, num_classes)
            
        if self.use_cas:
            self.uncertain_ratio = args['uncertain_ratio']
            self.over_sample_ratio = args['over_sample_ratio']
            self.num_samples = int(image_shape ** 3 * args['sample_ratio'])
            self.condition_phase = 'p2'
            self.cvae = ConditionalVAE((1,(128,128,128),1), self.image_shape, latent_dim=32)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.stem_features != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.stem_features,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.stem_features, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.stem_features = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.stem_features, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, input, mask=None, labels=None):
        res = {}
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        res['p2'] = x.clone()
        x = self.layer2(x)
        res['p3'] = x.clone()
        x = self.layer3(x)
        res['p4'] = x.clone()
        x = self.layer4(x)
        res['p5'] = x.clone()
        x = self.fpn(res)

        if self.training and self.use_cas:
            sample_coor, uncertain_sample_coor, _ = self.sample_points(input.clone())
            if labels is not None:
                labels = point_sample(
                    labels.unsqueeze(1).to(torch.float), sample_coor, mode="nearest", align_corners=False
                    ).squeeze(1).to(torch.long)
        else:
            sample_coor = None
        if self.do_vli:
            x = self.constract_logit(x, sample_coor=sample_coor, mask_index=mask)
        else:
            if self.training and self.use_cas:
                x = point_sample(x, sample_coor, align_corners=False).permute(0,2,1)
                transposed = False
            else:
                x = x.permute(0,2,3,4,1)
                transposed = True
            x = self.seg_out(x)
            if transposed:
                x = x.permute(0,4,1,2,3)
        
        if self.use_cas:
            if self.training:
                kwargs = {}
                point_indices, point_coords = get_uncertain_point_coords(
                        calculate_uncertainty_with_gt(x.permute(0,2,1),labels), 
                        (sample_coor.shape[1])//1, sample_coor=sample_coor, labels=labels)

                gaussian = generate_gaussian(self.image_shape, point_coords).unsqueeze(1).to(input.device)

                cvae_out = self.cvae(gaussian, input.clone())
                kwargs['recon_loss'] = nn.functional.mse_loss(cvae_out[0], cvae_out[1])
                mu, log_var = cvae_out[2].flatten(start_dim=1), cvae_out[3].flatten(start_dim=1)
                kwargs['kld_loss'] = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
                kwargs['sample_coor'] = sample_coor  
                kwargs['point_coords'] = point_coords  
                kwargs['uncertain_sample_coor'] = uncertain_sample_coor
                return x, labels, kwargs

        return x

    def constract_logit(self, x, u=0, sample_coor=None, mask_index=None):
        B, C, D, H, W = x.shape
        text_features = self.text_features.squeeze(dim=1) @ self.text_projection
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        if sample_coor is not None:
            image_features = point_sample(x, sample_coor, align_corners=False).permute(0,2,1)
        else:
            image_features = x.permute(0,2,3,4,1).contiguous().view(B*D*H*W, C)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        if mask_index is not None and sample_coor is None:
            image_features = image_features.gather(dim=0, index=mask_index.repeat(1,C))
        
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()

        if sample_coor is None and mask_index is None:
            logits_per_image = logits_per_image.view(B, D, H, W, -1).permute(0,4,1,2,3)
        return logits_per_image

    def sample_points(self, condition):
        num_samples = int(self.num_samples * self.over_sample_ratio)
        num_uncertain_points = int(self.uncertain_ratio * num_samples)
        num_random_points = num_samples - num_uncertain_points
        if num_uncertain_points > 0:
            gaussian = self.cvae.sample(condition.shape[0], condition)

            mask = torch.zeros_like(gaussian)
            mask[:,:,2:-2,2:-2,2:-2] = gaussian[:,:,2:-2,2:-2,2:-2]
            gaussian = mask
            _, uncertain_sample_coor = get_uncertain_point_coords_on_grid(gaussian, num_uncertain_points)
        else:
            uncertain_sample_coor = None

        if num_random_points > 0 and num_uncertain_points > 0:
            sample_coor = torch.cat(
                [uncertain_sample_coor, torch.rand(condition.shape[0], num_random_points, 3, device=condition.device)],
                dim=1,
            )
        elif num_random_points > 0 and num_uncertain_points == 0:
            sample_coor = torch.rand(condition.shape[0], num_random_points, 3, device=condition.device)
        elif num_random_points == 0 and num_uncertain_points > 0:
            sample_coor = uncertain_sample_coor
        return sample_coor, uncertain_sample_coor, gaussian

    def adjust_coor_on_grid(self, point_coor):
        """ adjust coordination onto grid so sampling will not need to interpolate """
        voxel_step = 1.0 / self.image_shape
        grid_coor = torch.clip(torch.round(point_coor * self.image_shape - 0.5),min=0,max=(self.image_shape-1)) + 0.5
        return voxel_step * grid_coor


class SemSegFPNHead(nn.Module):
    """
    A semantic segmentation head described in :paper:`PanopticFPN`
    """

    def __init__(
        self,
        input_shape,
        *,
        num_classes: int,
        conv_dims: int,
        common_stride: int,
        loss_weight: float = 1.0,
        norm = None,
        conv_op = None,
        ignore_value: int = -1,
    ):
        super().__init__()
        input_shape = input_shape.items()
        if not len(input_shape):
            raise ValueError("SemSegFPNHead(input_shape=) cannot be empty!")
        self.in_features = [k for k, v in input_shape]
        feature_strides = [v[-1] for k, v in input_shape]
        feature_channels = [v[0] for k, v in input_shape]

        self.ignore_value = ignore_value
        self.common_stride = common_stride
        self.loss_weight = loss_weight

        self.scale_heads = []
        for in_feature, stride, channels in zip(
            self.in_features, feature_strides, feature_channels
        ):
            head_ops = []
            head_length = max(1, int(np.log2(stride) - np.log2(self.common_stride)))
            for k in range(head_length):
                norm_module = norm(conv_dims)
                conv = conv_op(
                    channels if k == 0 else conv_dims,
                    conv_dims,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=not norm,
                )
                
                head_ops.append(nn.Sequential(conv,norm_module,nn.ReLU()))
                if stride != self.common_stride:
                    head_ops.append(
                        nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
                    )
            self.scale_heads.append(nn.Sequential(*head_ops))
            self.add_module(in_feature, self.scale_heads[-1])
        self.predictor = conv_op(conv_dims*len(self.in_features), num_classes, kernel_size=1, stride=1, padding=0)    # conv_dims*len(self.in_features)

    def forward(self, features, targets=None):
        res = []
        for i, f in enumerate(self.in_features):
            res.append(self.scale_heads[i](features[f]))
        res = torch.cat(res,dim=1)
        res = self.predictor(res)
        return res

def generate_gaussian(image_shape, sample_coor, visualize=False):
    B, N, _ = sample_coor.shape
    sample_coor = torch.clip(torch.round(sample_coor * image_shape - 0.5),min=0,max=(image_shape-1)).long()
    shift = torch.arange(B, dtype=torch.long, device=sample_coor.device).unsqueeze(1).unsqueeze(1).repeat(1, N, 1)
    sample_coor = einops.rearrange(torch.cat([shift, sample_coor],dim=-1),'b n c -> c (b n)')

    step = 1./float(N)
    upper, lower = 1.0, 0.0
    gaussian_sigma = 1.0
    gaussian = torch.zeros((B, image_shape, image_shape, image_shape)).float().index_put(tuple(sample_coor),(torch.arange(upper, lower, -((upper-lower)*step))).repeat(B))

    gaussian = torch.from_numpy(gaussian_filter(gaussian, [0, gaussian_sigma, gaussian_sigma, gaussian_sigma], 0, mode='constant', cval=0))
    return gaussian

def calculate_uncertainty_with_gt(logits, classes):
    if logits.shape[1] == 1:
        gt_class_logits = logits.clone()
    else:
        gt_class_logits = logits.gather(dim=1,index=classes.unsqueeze(1))
    return -(torch.abs(gt_class_logits))

def get_uncertain_point_coords_on_grid(uncertainty_map, num_points):
    B, _, D, H, W = uncertainty_map.shape
    d_step = 1.0 / float(D)
    h_step = 1.0 / float(H)
    w_step = 1.0 / float(W)

    num_points = min(D * H * W, num_points)
    point_indices = torch.topk(uncertainty_map.view(B, D * H * W), k=num_points, dim=1)[1]
    point_coords = torch.zeros(B, num_points, 3, dtype=torch.float, device=uncertainty_map.device)
    point_coords[:, :, 0] = d_step / 2.0 + torch.div(point_indices, (W*H), rounding_mode='trunc').to(torch.float) * w_step
    point_coords[:, :, 1] = h_step / 2.0 + torch.div(point_indices %(W*H), W, rounding_mode='trunc').to(torch.float) * h_step
    point_coords[:, :, 2] = w_step / 2.0 + (point_indices %(H*W)%W).to(torch.float) * w_step
    return point_indices, point_coords

def get_uncertain_point_coords(uncertainty_map, num_points, sample_coor, labels):
    B, _, K = uncertainty_map.shape
    num_points = min(K, num_points)
    allcase = []
    for idx in range(B):
        foreground_index = torch.arange(labels.shape[-1],device=labels.device)[labels[idx]!=0]
        background_index = torch.arange(labels.shape[-1],device=labels.device)[labels[idx]==0]
        fg_numpoint = min(foreground_index.shape[-1], num_points)
        bg_numpoint = min(background_index.shape[-1], (num_points - fg_numpoint))

        foreground_map = uncertainty_map[idx].view(K).gather(dim=0,index=foreground_index)
        map_indices = torch.topk(foreground_map, k=fg_numpoint)[1]
        point_indices = foreground_index[map_indices]
        
        if bg_numpoint > 0:
            background_map = uncertainty_map[idx].view(K).gather(dim=0,index=background_index)
            bg_map_indices = torch.topk(background_map, k=bg_numpoint)[1]
            point_indices = torch.cat([point_indices, background_index[bg_map_indices]], dim=0)
            map_indices = torch.cat([map_indices, bg_map_indices], dim=0)
        allcase.append(point_indices)
    point_indices = torch.stack(allcase, dim=0)
    point_coords = sample_coor.gather(dim=1,index=point_indices.unsqueeze(-1).expand(-1,-1,3))
    return point_indices, point_coords

def point_sample(input, point_coords, **kwargs):
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2).unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3).squeeze(3)
    return output
