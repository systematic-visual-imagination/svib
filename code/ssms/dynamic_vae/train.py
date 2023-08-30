import os
import git
import math
import argparse

from yacs.config import CfgNode

from datetime import datetime

import torch
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn import DataParallel as DP

import torchvision.utils as vutils

from ssms.dynamic_vae.modules import DynamicVAE
from ssms.dynamic_vae.data import GlobTaskVideoDataset

from helpers.annealing import *

from configs.dynamic_vae import dynamic_vae_configs


parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=40)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--steps', nargs='+', default=[100000, 200000, 300000, 400000])

parser.add_argument('--image_size', type=int, default=128)
parser.add_argument('--image_channels', type=int, default=3)

parser.add_argument('--checkpoint_path', default='checkpoint.pt.tar')
parser.add_argument('--data_path', default='data/')
parser.add_argument('--log_path', default='logs/')

parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--lr_warmup_steps', type=int, default=30000)
parser.add_argument('--lr_half_life', type=int, default=250000)
parser.add_argument('--clip', type=float, default=0.05)

parser.add_argument('--use_dp', default=True, action='store_true')

local_configs = parser.parse_args()

args = CfgNode()

args.update(dynamic_vae_configs)
args.update(vars(local_configs))

torch.manual_seed(args.seed)

arg_str_list = ['{}={}'.format(k, v) for k, v in vars(args).items()]
arg_str = '__'.join(arg_str_list)
log_dir = os.path.join(args.log_path, datetime.today().isoformat())
writer = SummaryWriter(log_dir)
print(f'log_dir = {log_dir}')
writer.add_text('hparams', arg_str)

repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha
writer.add_text('git', sha)


def visualize(video, video_recon, video_gen, N=8):
    video = video[:N, :, :, :, :]
    video_recon = video_recon[:N, :, :, :, :]
    video_gen = video_gen[:N, :, :, :, :]

    # tile
    tiles = torch.cat((video, video_recon, video_gen), dim=1).flatten(end_dim=1)

    # grid
    visual = vutils.make_grid(tiles, nrow=(tiles.shape[0] // N), pad_value=0.8)

    return visual


train_dataset = GlobTaskVideoDataset(root=args.data_path, phase='train', img_size=args.image_size)
val_dataset = GlobTaskVideoDataset(root=args.data_path, phase='val', img_size=args.image_size)

loader_kwargs = {
    'batch_size': args.batch_size,
    'shuffle': True,
    'num_workers': args.num_workers,
    'pin_memory': True,
    'drop_last': True,
}

train_loader = DataLoader(train_dataset, **loader_kwargs)
val_loader = DataLoader(val_dataset, **loader_kwargs)

train_epoch_size = len(train_loader)
val_epoch_size = len(val_loader)

log_interval = train_epoch_size // 5

model = DynamicVAE(args)

if os.path.isfile(args.checkpoint_path):
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    start_epoch = checkpoint['epoch']
    best_val_loss = checkpoint['best_val_loss']
    best_epoch = checkpoint['best_epoch']
    model.load_state_dict(checkpoint['model'])
else:
    checkpoint = None
    start_epoch = 0
    best_val_loss = math.inf
    best_epoch = 0

model = model.cuda()
if args.use_dp:
    model = DP(model)

optimizer = Adam([
    {'params': (x[1] for x in model.named_parameters()), 'lr': args.lr},
])
if checkpoint is not None:
    optimizer.load_state_dict(checkpoint['optimizer'])

for epoch in range(start_epoch, args.epochs):
    model.train()

    for batch, video in enumerate(train_loader):
        global_step = epoch * train_epoch_size + batch

        lr_warmup_factor = linear_warmup(
            global_step,
            0.,
            1.0,
            0.,
            args.lr_warmup_steps)

        lr_decay_factor = math.exp(global_step / args.lr_half_life * math.log(0.5))

        optimizer.param_groups[0]['lr'] = lr_decay_factor * lr_warmup_factor * args.lr

        video = video.cuda()

        optimizer.zero_grad()

        (loss, recon) = model(video)

        if args.use_dp:
            loss = loss.mean()

        loss.backward()
        clip_grad_norm_(model.parameters(), args.clip, 'inf')
        optimizer.step()

        with torch.no_grad():
            if batch % log_interval == 0:
                print('Train Epoch: {:3} [{:5}/{:5}] \t Loss: {:F}'.format(
                    epoch + 1, batch, train_epoch_size, loss.item()))

                writer.add_scalar('TRAIN/loss', loss.item(), global_step)
                writer.add_scalar('TRAIN/lr', optimizer.param_groups[0]['lr'], global_step)

    with torch.no_grad():
        gen = (model.module if args.use_dp else model).generate(video[:8, 0])
        visual = visualize(video, recon, gen, N=8)
        writer.add_image('TRAIN_recons/epoch={:03}'.format(epoch + 1), visual)

    with torch.no_grad():
        model.eval()

        val_loss = 0.

        for batch, video in enumerate(val_loader):
            video = video.cuda()

            (loss, recon) = model(video)

            if args.use_dp:
                loss = loss.mean()

            val_loss += loss.item()

        val_loss /= (val_epoch_size)

        writer.add_scalar('VAL/loss', val_loss, epoch + 1)

        print('====> Epoch: {:3} \t Loss = {:F}'.format(epoch + 1, val_loss))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1

            torch.save(model.module.state_dict() if args.use_dp else model.state_dict(),
                       os.path.join(log_dir, 'best_model.pt'))

            for steps in args.steps:
                if global_step < steps:
                    torch.save(model.module.state_dict() if args.use_dp else model.state_dict(),
                               os.path.join(log_dir, f'best_model_until_{steps}_steps.pt'))

            if 50 <= epoch:
                gen = (model.module if args.use_dp else model).generate(video[:8, 0])
                visual = visualize(video, recon, gen, N=8)
                writer.add_image('VAL_recons/epoch={:03}'.format(epoch + 1), visual)

        writer.add_scalar('VAL/best_loss', best_val_loss, epoch + 1)

        checkpoint = {
            'epoch': epoch + 1,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'model': model.module.state_dict() if args.use_dp else model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        torch.save(checkpoint, os.path.join(log_dir, 'checkpoint.pt.tar'))

        print('====> Best Loss = {:F} @ Epoch {}'.format(best_val_loss, best_epoch))

writer.close()
