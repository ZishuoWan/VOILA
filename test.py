import os
import torch
from utils.data_loader import CustomDataset
import argparse
from torch.utils.data import DataLoader
from utils.utils import calculate_metric_percase
import numpy as np
from utils.class_map import class_name
from model.voila import VOILA
from utils.predict_with_sliding_window import nnUNetPredictor
import json

def convertCheckpoint(state_dict):
    new_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            new_dict[''.join(k.split('module.'))] = v
        else:
            new_dict[k] = v
    return new_dict

def test(args):
    device = torch.device('cuda')
    if args.use_vli:
        text_features = torch.load(args.text_features, map_location=device)
    else:
        text_features = None
    
    model = VOILA(
        num_classes = args.num_classes, stem_features=args.stem_features, text_features=text_features,
        image_shape=args.image_shape, use_cas=args.cas,uncertain_ratio=0.,sample_ratio=0., over_sample_ratio=0.
        ).to(device)
    
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        temp_state_dict = convertCheckpoint(checkpoint['model'])
        model.load_state_dict(temp_state_dict)
        for n, p in model.named_parameters():
            p.requires_grad = False

    model.eval()
    dataset = CustomDataset('test', args.data_path, args.basePath, args.image_shape)
    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False) 
    
    metric_dict = {}
    voxel_dict = {}
    args.data_len = len(test_dataloader)
    slidingPredictor = nnUNetPredictor(
        patch_size=args.image_shape,network=model, device=device, 
        num_segmentation_heads=args.num_classes,
        perform_everything_on_gpu=perform_everything_on_gpu
        )

    with torch.no_grad():
        for iter_idx, (images, labels, kwargs) in enumerate(test_dataloader):
            print(f'Predicting {iter_idx+1} / {args.data_len}.')
            out = slidingPredictor.predict_sliding_window_return_logits(images)
            out = out.squeeze(0).cpu().detach().numpy()
            labels = labels.squeeze(0).squeeze(0).cpu().detach().numpy()
            metric_i = {}
            voxel_i = {}
            for i, class_idx in enumerate(list(np.arange(1,118))):
                if class_idx not in metric_i.keys():
                    metric_i[class_idx] = []
                    voxel_i[class_idx] = []
                metric_i[class_idx].append(calculate_metric_percase(out == class_idx, labels == class_idx))
                voxel_i[class_idx].append(np.sum(labels == class_idx) / 1000.)

    
    allclass_dice = []
    allclass_hd95 = []
    for k, v in metric_dict.items():
        if k == 0: continue
        v = np.stack(v)
        mean_dice = np.mean(v, axis=0)[0, 0]
        mean_hd95 = np.mean(v, axis=0)[0, 1]
        mean_voxel = np.mean(voxel_dict[k])
        allclass_dice.append(mean_dice)
        allclass_hd95.append(mean_hd95)
        print(f'Mean dice: {mean_dice}, mean hd95: {mean_hd95}.')
        with open(args.txt_path, 'a') as f:
            f.write(f'Organ: {class_name[k]}, mean dice: {mean_dice}, mean hd95: {mean_hd95}, mean voxel: {mean_voxel*1000}')
            f.write('\n')
            f.flush()
            f.close()
    with open(args.txt_path, 'a') as f:
            f.write(f'Total, mean dice: {np.mean(allclass_dice)}, mean hd95: {np.mean(allclass_hd95)}.')
            f.write(f'\n# {args.checkpoint}\t\n\n')
            f.flush()
            f.close()



perform_everything_on_gpu = True
patch_pred = True
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_arch", default=None)

    parser.add_argument("--device")
    parser.add_argument("--data_len", default=0, help='len(dataloader)')
    args = parser.parse_args()
    return args

def main(check_point):
    args = get_args()
    if check_point.endswith('epoch_last.pt'):
        param_file = '/'.join(check_point.split('/')[:-1]) + '/params.json'
    else:
        param_file = '/'.join(check_point.split('/')[:-2]) + '/params.json'
    args_dict = vars(args)
    with open(param_file, 'r') as f:
        args_dict.update(json.load(f))
    args.data_path = f'data/test_totalseg.csv'
    args.checkpoint = check_point
    args.txt_path = f'totalseg_{args.model_arch}_patchpred.txt'
    test(args)


if __name__ == '__main__':
    check_point = os.path.join(f'')
    main(check_point)
