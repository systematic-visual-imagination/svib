import os
import json

import torch
import argparse
import lpips

from torch.utils.data import DataLoader
import torchvision.utils as vutils

from yacs.config import CfgNode

from datetime import datetime

from data import GlobSysVImDataset

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=40)
parser.add_argument('--max_images', type=int, default=300)
parser.add_argument('--num_workers', type=int, default=4)

parser.add_argument('--image_size', type=int, default=128)
parser.add_argument('--image_channels', type=int, default=3)

parser.add_argument('--data_path', default='data/')
parser.add_argument('--data_phase', default='full')

parser.add_argument('--representer', default='slot_attn')
parser.add_argument('--predictor', default='image_transformer')
parser.add_argument('--load_path', default='saved_models.pt.tar')

parser.add_argument('--num_slots', type=int, default=4)

args_script = parser.parse_args()

args_representer = CfgNode()


if args_script.representer == 'cnn_single_vector':
    from configs.cnn_single_vector import cnn_single_vector_configs
    args_representer.update(cnn_single_vector_configs)
elif args_script.representer == 'vit':
    from configs.vit import vit_configs
    args_representer.update(vit_configs)
elif args_script.representer == 'dynamic_vae':
    from configs.dynamic_vae import dynamic_vae_configs
    args_representer.update(dynamic_vae_configs)
elif args_script.representer == 'dynamic_slot':
    from configs.dynamic_slot import dynamic_slot_configs
    args_representer.update(dynamic_slot_configs)
else:
    raise NotImplementedError

args_representer.update(vars(args_script))

args_predictor = CfgNode()

if args_script.predictor == 'patch_transformer':
    from configs.patch_transformer import patch_transformer_configs
    args_predictor.update(patch_transformer_configs)
else:
    raise NotImplementedError

args_predictor.update(vars(args_script))

args_predictor.__representation_size__ = args_representer.__representation_size__

# Setup Seed
torch.manual_seed(args_script.seed)

test_dataset = GlobSysVImDataset(root=args_script.data_path, phase=args_script.data_phase, img_size=args_script.image_size)

loader_kwargs = {
    'batch_size': args_script.batch_size,
    'shuffle': False,
    'num_workers': args_script.num_workers,
    'pin_memory': True,
    'drop_last': True,
}

test_loader = DataLoader(test_dataset, **loader_kwargs)

test_epoch_size = len(test_loader)

# Setup Models
if args_script.representer == 'cnn_single_vector':
    from representers.joint.cnn import CNNSingleVectorEncoder
    model_representer = CNNSingleVectorEncoder(args_representer)
elif args_script.representer == 'cnn_relational':
    from representers.joint.cnn import CNNRelationalEncoder
    model_representer = CNNRelationalEncoder(args_representer)
elif args_script.representer == 'vit':
    from representers.joint.vit import ViTEncoder
    model_representer = ViTEncoder(args_representer)
elif args_script.representer == 'dynamic_vae':
    from ssms.dynamic_vae import NextFrameLatentPredictorVAE
    model_representer = NextFrameLatentPredictorVAE(args_representer)
elif args_script.representer == 'dynamic_slot':
    from ssms.dynamic_slot import NextFrameLatentPredictorSlot
    model_representer = NextFrameLatentPredictorSlot(args_representer)
else:
    raise NotImplementedError

if args_script.predictor == 'patch_transformer':
    from predictors.patch_transformer import PatchTransformer
    model_predictor = PatchTransformer(args_predictor)
else:
    raise NotImplementedError

# Setup GPU
saved_weights = torch.load(args_script.load_path, map_location='cpu')

model_representer.load_state_dict(saved_weights['model_representer'])
model_predictor.load_state_dict(saved_weights['model_predictor'])

model_representer = model_representer.cuda()
model_predictor = model_predictor.cuda()

# Evaluate
dump_dir = f"{args_script.load_path}_eval_{datetime.today().isoformat()}"
os.makedirs(dump_dir)


with torch.no_grad():
    model_representer.eval()
    model_predictor.eval()

    lpips_fn = lpips.LPIPS(net='alex', version='0.1')
    lpips_fn.cuda()

    test_mse = 0.
    test_lpips = 0.
    num_seen = 0
    for batch, (source_image, target_image) in enumerate(test_loader):
        B, *_ = source_image.shape

        source_image = source_image.cuda()
        target_image = target_image.cuda()

        source_z = model_representer(source_image)
        pred_image = model_predictor.decode(source_z)

        mse = ((pred_image - target_image) ** 2).sum()
        lpips_score = lpips_fn.forward(2 * pred_image - 1, 2 * target_image - 1).sum()

        test_mse += mse
        test_lpips += lpips_score
        num_seen += B

        print(f'Images Seen {num_seen:05d} MSE = {test_mse / num_seen} \t LPIPS = {test_lpips / num_seen}')

        if batch == 0:
            vutils.save_image(
                vutils.make_grid(source_image[:16], nrow=4, pad_value=0.8),
                os.path.join(dump_dir, 'source.png'))

            vutils.save_image(
                vutils.make_grid(target_image[:16], nrow=4, pad_value=0.8),
                os.path.join(dump_dir, 'target.png'))

            vutils.save_image(
                vutils.make_grid(pred_image[:16], nrow=4, pad_value=0.8),
                os.path.join(dump_dir, 'predicted.png'))

        if num_seen >= args_script.max_images:
            break

    test_mse /= num_seen
    test_lpips /= num_seen

    print(f'====> Overall MSE = {test_mse} \t LPIPS = {test_lpips}')

    with open(os.path.join(dump_dir, 'result.json'), "w") as fp:
        result = {}
        result['args_script'] = dict(vars(args_script))
        result['args_representer'] = dict(args_representer)
        result['args_predictor'] = dict(args_predictor)

        # add results
        result['mse'] = test_mse.item()
        result['lpips'] = test_lpips.item()

        json.dump(result, fp)
