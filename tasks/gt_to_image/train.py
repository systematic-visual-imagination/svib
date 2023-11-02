import git
import argparse

from yacs.config import CfgNode

from datetime import datetime

from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn import DataParallel as DP

import torchvision.utils as vutils

from helpers.annealing import *

from data import *

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=40)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--epochs', type=int, default=500)

parser.add_argument('--save_model_every_steps', type=int, default=20000)

parser.add_argument('--image_size', type=int, default=128)
parser.add_argument('--image_channels', type=int, default=3)

parser.add_argument('--data_path', default='data/')
parser.add_argument('--data_reader', default='dsprites')

parser.add_argument('--log_path', default='logs/')

parser.add_argument('--joint', default=True, action='store_true')
parser.add_argument('--representer', default='gt')
parser.add_argument('--representer_path', default=None)
parser.add_argument('--predictor', default='patch_transformer')

parser.add_argument('--lr_representer', type=float, default=1e-4)
parser.add_argument('--lr_predictor', type=float, default=3e-4)
parser.add_argument('--lr_warmup_steps', type=int, default=30000)
parser.add_argument('--lr_half_life', type=int, default=250000)
parser.add_argument('--clip', type=float, default=0.05)

parser.add_argument('--use_dp', default=True, action='store_true')

# Setup Configurations
args_script = parser.parse_args()

args_representer = CfgNode()

if args_script.representer == 'gt':
    from configs.gt import gt
    args_representer.update(gt)
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

# Setup Logging
log_dir = os.path.join(args_script.log_path, datetime.today().isoformat())
writer = SummaryWriter(log_dir)
print(f'log_dir = {log_dir}')

writer.add_text('args_representer', '\n'.join(['{}={}'.format(k, v) for k, v in dict(args_representer).items()]))
writer.add_text('args_predictor', '\n'.join(['{}={}'.format(k, v) for k, v in dict(args_predictor).items()]))

repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha
writer.add_text('git', sha)

# Setup Datasets
if args_script.data_reader == 'clevr':
    train_dataset = GlobSysVImDataset_GT_CLEVR(root=args_script.data_path, phase='train', img_size=args_script.image_size)
    val_dataset = GlobSysVImDataset_GT_CLEVR(root=args_script.data_path, phase='val', img_size=args_script.image_size)
elif args_script.data_reader == 'clevrtex':
    train_dataset = GlobSysVImDataset_GT_CLEVRTex(root=args_script.data_path, phase='train', img_size=args_script.image_size)
    val_dataset = GlobSysVImDataset_GT_CLEVRTex(root=args_script.data_path, phase='val', img_size=args_script.image_size)
elif args_script.data_reader == 'dsprites':
    train_dataset = GlobSysVImDataset_GT_DSPRITES(root=args_script.data_path, phase='train', img_size=args_script.image_size)
    val_dataset = GlobSysVImDataset_GT_DSPRITES(root=args_script.data_path, phase='val', img_size=args_script.image_size)
else:
    raise NotImplementedError

loader_kwargs = {
    'batch_size': args_script.batch_size,
    'shuffle': True,
    'num_workers': args_script.num_workers,
    'pin_memory': True,
    'drop_last': True,
}

train_loader = DataLoader(train_dataset, **loader_kwargs)
val_loader = DataLoader(val_dataset, **loader_kwargs)

train_epoch_size = len(train_loader)
val_epoch_size = len(val_loader)

log_interval = train_epoch_size // 5

# Setup Models
if args_script.representer == 'gt':
    from representers.gt import GTEncoder
    model_representer = GTEncoder(args_representer)
else:
    raise NotImplementedError

if args_script.predictor == 'patch_transformer':
    from predictors.patch_transformer import PatchTransformer
    model_predictor = PatchTransformer(args_predictor)
else:
    raise NotImplementedError

# Load Pre-trained Representer
if args_script.representer_path:
    model_representer.load_state_dict(torch.load(args_script.representer_path, map_location='cpu'))

# Setup GPU
model_representer = model_representer.cuda()
model_predictor = model_predictor.cuda()
if args_script.use_dp:
    model_representer = DP(model_representer)
    model_predictor = DP(model_predictor)

# Setup Monitors
checkpoint = None
start_epoch = 0
best_val_loss = math.inf
best_epoch = 0

optimizer = Adam([
    {'params': (x[1] for x in model_representer.named_parameters()), 'lr': args_script.lr_representer},
    {'params': (x[1] for x in model_predictor.named_parameters()), 'lr': args_script.lr_predictor},
])

for epoch in range(start_epoch, args_script.epochs):
    model_representer.train()
    model_predictor.train()

    for batch, (discrete_factors, float_factors, target_image) in enumerate(train_loader):
        global_step = epoch * train_epoch_size + batch

        if args_script.predictor == 'image_transformer':
            tau = cosine_anneal(
                    global_step,
                    args_predictor.tau_start,
                    args_predictor.tau_final,
                    0,
                    args_predictor.tau_steps)
            predictor_kwargs = {'tau': tau}
        elif args_script.predictor == 'patch_transformer':
            predictor_kwargs = {}
        else:
            raise NotImplementedError

        lr_warmup_factor = linear_warmup(
            global_step,
            0.,
            1.0,
            0,
            args_script.lr_warmup_steps)

        lr_decay_factor = math.exp(global_step / args_script.lr_half_life * math.log(0.5))

        optimizer.param_groups[0]['lr'] = lr_decay_factor * lr_warmup_factor * args_script.lr_representer
        optimizer.param_groups[1]['lr'] = lr_decay_factor * lr_warmup_factor * args_script.lr_predictor

        discrete_factors = discrete_factors.cuda()
        float_factors = float_factors.cuda()
        target_image = target_image.cuda()

        optimizer.zero_grad()

        if args_script.joint:
            source_z = model_representer(discrete_factors, float_factors)
        else:
            with torch.no_grad():
                source_z = model_representer(discrete_factors, float_factors)

        loss = model_predictor(source_z, target_image, **predictor_kwargs)

        if args_script.use_dp:
            loss = loss.mean()

        loss.backward()

        clip_grad_norm_(model_representer.parameters(), args_script.clip, 'inf')
        clip_grad_norm_(model_predictor.parameters(), args_script.clip, 'inf')

        optimizer.step()

        with torch.no_grad():
            if batch % log_interval == 0:
                print('Train Epoch: {:3} [{:5}/{:5}] \t Loss: {:F}'.format(
                    epoch + 1, batch, train_epoch_size, loss.item()))

                writer.add_scalar('TRAIN/loss', loss.item(), global_step)
                writer.add_scalar('TRAIN/lr_representer', optimizer.param_groups[0]['lr'], global_step)
                writer.add_scalar('TRAIN/lr_predictor', optimizer.param_groups[1]['lr'], global_step)
            if global_step % args_script.save_model_every_steps == 0:
                checkpoint = {
                    'global_step': global_step,
                    'model_representer': model_representer.module.state_dict() if args_script.use_dp else model_representer.state_dict(),
                    'model_predictor': model_predictor.module.state_dict() if args_script.use_dp else model_predictor.state_dict(),
                }
                torch.save(checkpoint, os.path.join(log_dir, f'saved_models_{global_step:08d}.pt.tar'))


    with torch.no_grad():
        writer.add_image('TRAIN_target/epoch={:03}'.format(epoch + 1),
                         vutils.make_grid(target_image[:16], nrow=4, pad_value=0.8))

        recons = (model_predictor.module if args_script.use_dp else model_predictor).decode(source_z[:16])
        writer.add_image('TRAIN_recons/epoch={:03}'.format(epoch + 1),
                         vutils.make_grid(recons, nrow=4, pad_value=0.8))

    with torch.no_grad():
        model_representer.eval()
        model_predictor.eval()

        val_loss = 0.

        for batch, (discrete_factors, float_factors, target_image) in enumerate(val_loader):
            discrete_factors = discrete_factors.cuda()
            float_factors = float_factors.cuda()
            target_image = target_image.cuda()

            source_z = model_representer(discrete_factors, float_factors)
            loss = model_predictor(source_z, target_image, **predictor_kwargs)

            if args_script.use_dp:
                loss = loss.mean()

            val_loss += loss.item()

        val_loss /= val_epoch_size

        writer.add_scalar('VAL/loss', val_loss, epoch + 1)

        print('====> Epoch: {:3} \t Loss = {:F}'.format(epoch + 1, val_loss))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1

            torch.save(model_representer.module.state_dict() if args_script.use_dp else model_representer.state_dict(),
                       os.path.join(log_dir, 'best_model_representer.pt'))
            torch.save(model_predictor.module.state_dict() if args_script.use_dp else model_predictor.state_dict(),
                       os.path.join(log_dir, 'best_model_predictor.pt'))

        writer.add_scalar('VAL/best_loss', best_val_loss, epoch + 1)

        checkpoint = {
            'epoch': epoch + 1,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'model_representer': model_representer.module.state_dict() if args_script.use_dp else model_representer.state_dict(),
            'model_predictor': model_predictor.module.state_dict() if args_script.use_dp else model_predictor.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        torch.save(checkpoint, os.path.join(log_dir, 'checkpoint.pt.tar'))
        print('====> Best Loss = {:F} @ Epoch {}'.format(best_val_loss, best_epoch))

writer.close()
