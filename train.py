import argparse
import json
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.distributed
from apex import amp
from apex.parallel import DistributedDataParallel as DDP
from apex.parallel import convert_syncbn_model
from tensorboardX import SummaryWriter
from torch import nn, optim
from monai.data import dataloader

from model.voila import VOILA, point_sample
from utils.data_loader import CustomDataset
from utils.utils import CEforPixelContrast, VoxelF1Loss
import time

SEED = 42
def setup_seed(seed):
    torch.manual_seed(seed)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        # torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # cudnn.deterministic = True
setup_seed(SEED)

def convertCheckpoint(state_dict):
    new_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            new_dict[''.join(k.split('module.'))] = v
        else:
            new_dict[k] = v
    return new_dict

def maybe_SaveModel(args, checkpoint):
    if args.epoch % args.save_interval == 0:
        torch.save(checkpoint, os.path.join(args.save_path, 'pt', f'epoch_{args.epoch}.pt'))
    if args.epoch == args.max_epoch:
        torch.save(checkpoint, os.path.join(args.save_path, 'epoch_last.pt'))

def AfterIter(args, **kwargs):
    if args.local_rank == 0:
        args.batch_losses.append(kwargs['loss']['total'])
        args.batch_celosses.append(kwargs['loss']['ce'])
        if args.use_dice: args.batch_dclosses.append(kwargs['loss']['dc'])
        if args.cas: 
            args.batch_reconlosses.append(kwargs['loss']['recon_loss'])
            args.batch_kldlosses.append(kwargs['loss']['kld_loss'])
        args.batch_dice = kwargs['dice']
        print(f"Batch {kwargs['iter_idx']} / {args.data_len}, loss = {args.batch_losses[-1]}, dice, data time:{kwargs['datatime']-kwargs['starttime']}, implement time:{kwargs['impletime']-kwargs['datatime']}, bptime:{kwargs['bptime']-kwargs['impletime']}")


def AfterEpoch(args, **kwargs):
    if args.local_rank == 0:
        maybe_SaveModel(args, kwargs['checkpoint'])
        epoch_loss = np.mean(args.batch_losses)
        epoch_celoss = np.mean(args.batch_celosses)
        
        epoch_dice = np.mean(args.batch_dice)
        args.batch_losses = []
        args.batch_celosses = []
        args.batch_dice = []
        if args.use_dice: 
            epoch_dcloss = np.mean(args.batch_dclosses)
            args.batch_dclosses = []
            args.log.add_scalar('dc_loss', epoch_dcloss, args.epoch)
        if args.cas: 
            epoch_reconloss = np.mean(args.batch_reconlosses)
            args.batch_reconlosses = []
            args.log.add_scalar('recon_loss', epoch_reconloss, args.epoch)
            epoch_kldloss = np.mean(args.batch_kldlosses)
            args.batch_kldlosses = []
            args.log.add_scalar('kld_loss', epoch_kldloss, args.epoch)
        args.log.add_scalar('total_loss', epoch_loss, args.epoch)
        args.log.add_scalar('ce_loss', epoch_celoss, args.epoch)
        print(f'Epoch {args.epoch} / {args.max_epoch}, loss = {epoch_loss}, dice = {epoch_dice}')

def maybe_InitSavePath(args):
    if args.local_rank == 0:
        if not args.debug:
            args.save_path = os.path.join(os.getcwd(), f'checkpoint/{args.dataset_id}', datetime.now().strftime('model_%Y%m%d_%H%M%S'))
        else:
            args.save_path = os.path.join(os.getcwd(), 'debug/checkpoint', datetime.now().strftime('model_%Y%m%d_%H%M%S'))
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        if not os.path.exists(os.path.join(args.save_path, 'pt')):
            os.makedirs(os.path.join(args.save_path, 'pt'))
        if not os.path.exists(os.path.join(args.save_path, 'sample_point')):
            os.makedirs(os.path.join(args.save_path, 'sample_point'))

        with open(f"{args.save_path}/params.json", mode="w") as f:
            json.dump(args.__dict__, f, indent=4)
        args.log = SummaryWriter(args.save_path+'/tfwriter_train')
        
def train(args):
    device = torch.device('cuda:{}'.format(args.local_rank))
    if args.use_vli:
        text_features = torch.load(args.text_features, map_location=device)
    else:
        text_features = None
        
    model = VOILA(
        num_classes=args.num_classes, stem_features=args.stem_features, text_features=text_features,
        image_shape=args.image_shape, use_cas=args.cas,
        uncertain_ratio=args.uncertain_ratio,sample_ratio=args.sample_ratio, over_sample_ratio=args.over_sample_ratio,
        ).to(device)
    
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        temp_state_dict = convertCheckpoint(checkpoint['model'])

        for k, v in model.named_parameters():
            if k not in temp_state_dict.keys():
                print(f'Adding {k} into the checkpoint.')
                temp_state_dict[k] = v
        model.load_state_dict(temp_state_dict)
        torch.cuda.empty_cache()
    
        
    if args.local_rank == 0:
        print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    
    if args.use_amp: model = convert_syncbn_model(model)

    dataset = CustomDataset(args.dl_mode, args.data_path, args.basePath, args.image_shape, deep_supervise=False, args=args)
    if args.use_amp:
        train_sampler  = torch.utils.data.distributed.DistributedSampler(dataset)
        train_dataloader = dataloader.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, sampler=train_sampler, pin_memory=False, drop_last=True)
    else:
        train_dataloader = dataloader.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

    criterion = CEforPixelContrast(num_classes=118, ignore_index=args.ignore_index)
    if args.use_dice: dice_loss = VoxelF1Loss(True, True, False, 1e-5, True)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-5)
    if args.checkpoint is not None: 
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except:
            pass

    if args.use_amp:
        amp.register_float_function(torch, 'sigmoid')
        amp.register_float_function(torch, 'softmax')
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        if args.checkpoint is not None: amp.load_state_dict(checkpoint['amp'])
        model = DDP(model, delay_allreduce=True)

    args.data_len = len(train_dataloader)

    while args.epoch <= args.max_epoch:
        starttime = time.time()
        for iter_idx, (images, labels, kwargs) in enumerate(train_dataloader):
            datatime = time.time()
            images = images.to(device)
            if isinstance(labels, (list, tuple)): 
                labels = [i.to(device, non_blocking=True) for i in labels]
            else:    
                labels = labels.to(device).long()

            if args.cas:
                outputs = model(images, None, labels)
                output, sample_labels, kwargs = outputs
                output = output.view(-1, output.shape[-1])
                sample_coor = kwargs['sample_coor']                         # all sampled voxels
                uncertain_sample_coor = kwargs['uncertain_sample_coor']     # voxels sampled by CAS
                recon_loss = kwargs['recon_loss']
                kld_loss = kwargs['kld_loss']
                if sample_labels is None:
                    labels = point_sample(
                        labels.unsqueeze(1).to(torch.float), sample_coor, mode="nearest", align_corners=False
                        ).squeeze(1).to(torch.long).flatten()
                else:
                    labels = sample_labels.flatten()
            
            else:
                output = model(images)
            loss_ce, dice = criterion(output, labels)
            if args.use_dice:
                loss_dc = dice_loss(output, labels)
                total_loss = loss_ce + loss_dc
            else:
                total_loss = loss_ce
            if args.cas:
                if args.epoch % 1 == 0:
                    total_loss = total_loss + recon_loss + args.kld_weight * kld_loss
                else:
                    pass
            impletime = time.time()
            
            optimizer.zero_grad()
            if args.use_amp:
                with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                    scaled_loss.backward() 
            else: total_loss.backward()
            optimizer.step()
            bptime = time.time()
            
            if args.use_amp: 
                torch.distributed.all_reduce(total_loss) 
                torch.distributed.all_reduce(loss_ce)
                if args.use_dice: torch.distributed.all_reduce(loss_dc)
                if args.cas: 
                    torch.distributed.all_reduce(recon_loss)
                    torch.distributed.all_reduce(kld_loss)
            if args.use_dice: 
                losses = {
                    'total': total_loss.detach().cpu().numpy()/args.nproc,
                    'ce': loss_ce.detach().cpu().numpy()/args.nproc,
                    'dc': loss_dc.detach().cpu().numpy()/args.nproc
                }
            else:
                losses = {
                    'total': total_loss.detach().cpu().numpy()/args.nproc,
                    'ce': loss_ce.detach().cpu().numpy()/args.nproc,
                }
            if args.cas:
                losses['recon_loss'] = recon_loss.detach().cpu().numpy()/args.nproc
                losses['kld_loss'] = kld_loss.detach().cpu().numpy()/args.nproc

            AfterIter(args, loss=losses, iter_idx=iter_idx+1, dice=dice.detach().cpu().numpy(),\
                starttime=starttime,datatime=datatime,impletime=impletime,bptime=bptime)
            starttime = time.time()

        if args.use_amp:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'amp': amp.state_dict()
            }
        else:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
        AfterEpoch(args, checkpoint=checkpoint)
        args.epoch += 1


# To use apex, please use: CUDA_VISIBLE_DEVICES=1,7,5,6 python -m torch.distributed.run --nproc_per_node=4 train.py --use_amp
def get_args():
    parser = argparse.ArgumentParser()
    ## for distributed training
    parser.add_argument("--model_arch", default='voila')
    parser.add_argument("--stem_features", default=32)
    parser.add_argument("--nproc", default=2)
    parser.add_argument("--dataset_id", default='totalseg')
    parser.add_argument("--max_epoch", default=400)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--use_dice", default=True)
    parser.add_argument("--lr", default=1e-4)
    parser.add_argument("--deep_supervise", default=False)
    parser.add_argument("--use_fpn", default=True)
    parser.add_argument("--ignore_index", type=int, default=-100)       # -100 or 0
    parser.add_argument("--use_vli", default=True, help='use voxel-language interaction or not')
    parser.add_argument("--num_classes", default=118)
    parser.add_argument("--image_shape", default=128)
    parser.add_argument("--cas", default=True, help='use conplexity-aware sampling or not')
    parser.add_argument("--uncertain_ratio", type=float, default=0.5)
    parser.add_argument("--over_sample_ratio", type=float, default=2.)     # sampling number = DHW * sample_ratio * over_sample_ratio
    parser.add_argument("--sample_ratio", type=float, default=0.1)        # CAS sampling number = sampling number * uncertain_ratio
    parser.add_argument("--kld_weight", type=float, default=1e-8)
    parser.add_argument("--basePath", type=str, help='where to find the data')

    # No need to change the params below.
    parser.add_argument("--batch_size", default=2, help='batch size per GPU')
    parser.add_argument("--text_features", default='text_features.pt')
    parser.add_argument("--save_interval", default=50)
    parser.add_argument("--num_workers", default=2)
    parser.add_argument("--dl_mode", default='train')   
    parser.add_argument('--local_rank', type=int, default=os.getenv('LOCAL_RANK', -1))
    parser.add_argument("--device")
    parser.add_argument("--use_amp", action='store_true', default=False)
    parser.add_argument("--epoch", default=1)
    parser.add_argument("--save_path", default=None)
    parser.add_argument("--log", default=None, help='tensorboard SummaryWriter')
    parser.add_argument("--seed", default=SEED)
    parser.add_argument("--batch_losses", default=[], help='losses in one batch')
    parser.add_argument("--batch_celosses", default=[], help='ce losses in one batch')
    parser.add_argument("--batch_dclosses", default=[], help='dc losses in one batch')
    parser.add_argument("--batch_reconlosses", default=[], help='recon losses in one batch')
    parser.add_argument("--batch_kldlosses", default=[], help='kld losses in one batch')
    parser.add_argument("--batch_outputs", default={'logit_I': [], 'logit_T':[]}, help='outputs in one batch, to calculate topk')
    parser.add_argument("--batch_targets", default=[], help='targets in one batch, to calculate topk')
    parser.add_argument("--batch_dice", default=[], help='mean dice score in one batch, to calculate mean for all')
    parser.add_argument("--batch_hd95", default=[], help='mean hd95 in one batch, to calculate mean for all')
    parser.add_argument("--data_len", default=0, help='len(dataloader)')
    parser.add_argument("--debug", action='store_true', default=False)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    args.data_path = f'data/train_{args.dataset_id}.csv'

    torch.cuda.set_device(args.local_rank)
    if args.use_amp:
        torch.distributed.init_process_group(
            'nccl',
            init_method='env://'
        )
    maybe_InitSavePath(args)
    train(args)

if __name__ == '__main__':
    main()
