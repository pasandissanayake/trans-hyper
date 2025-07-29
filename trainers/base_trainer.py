"""
    A basic trainer.

    Define train_step(self, data) -> metrics (e.g.: loss, accuracy) and 
    evaluate_step(self, data) -> metrics according to the specific cases

    The general procedure in run() is:
        make_datasets()
            create . train_loader, test_loader, dist_samplers
        make_model()
            create . model_ddp, model
        train()
            create . optimizer, epoch, log_buffer
            for epoch = 1 ... max_epoch:
                adjust_learning_rate()
                train_epoch()
                    train_step()
                evaluate_epoch()
                    evaluate_step()
                visualize_epoch()
                save_checkpoint()
"""

import os
import os.path as osp
from abc import ABC, abstractmethod
import time
import yaml
import wandb
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

import models
import utils
from trainers import register

TRAINER_NAME = "base_trainer"

@register(TRAINER_NAME)
class BaseTrainer(ABC):

    def __init__(self, rank, cfg, train_ds=None, test_ds=None):
        self.name = TRAINER_NAME
        self.rank = rank
        self.cfg = cfg
        self.trainer_cfg = self.cfg.trainer
        self.debug = self.cfg.debug() or self.cfg.debug_trainer()

        self.train_ds = train_ds
        self.test_ds = test_ds

        self.is_master = (rank == 0)
        
        self.total_gpus = self.cfg.env.total_gpus()
        self.distributed = (self.total_gpus > 1)

        # Setup log, tensorboard, wandb
        if self.is_master:
            logger, writer = utils.set_save_dir(self.cfg.env.save_dir(), replace=False)
            with open(osp.join(self.cfg.env.save_dir(), 'cfg.yaml'), 'w') as f:
                yaml.dump(cfg, f, sort_keys=False)

            self.log = logger.info

            self.enable_tb = True
            self.writer = writer

            if self.cfg.env.wandb_upload():
                self.enable_wandb = True
                # with open(cfg.wandb_cfg(), 'r') as f:
                #     wandb_cfg = yaml.load(f, Loader=yaml.FullLoader)
                # os.environ['WANDB_DIR'] = self.cfg.env.save_dir()
                # os.environ['WANDB_NAME'] = self.cfg.env.exp_name()
                # os.environ['WANDB_API_KEY'] = wandb_cfg['api_key']
                # wandb.init(project=wandb_cfg['project'], entity=wandb_cfg['entity'], config=cfg)
            else:
                self.enable_wandb = False
        else:
            self.log = lambda *args, **kwargs: None
            self.enable_tb = False
            self.enable_wandb = False

        # Setup distributed devices
        torch.cuda.set_device(rank)
        self.device = torch.device('cuda', torch.cuda.current_device())

        if self.distributed:
            dist_url = f"tcp://localhost:{self.cfg.env.port()}"
            dist.init_process_group(backend='nccl', init_method=dist_url,
                                    world_size=self.total_gpus, rank=rank)
            self.log(f'Distributed training enabled.')

        cudnn.benchmark = self.cfg.env.cudnn()

        self.log(f'Environment setup done.')

    def run(self):
        self.make_datasets()

        if self.cfg.eval_model():
            checkpoint = torch.load(self.cfg.eval_model())
            print(checkpoint.keys())
            cfg = utils.Config(cfg_dict=checkpoint['cfg'])
            self.make_model(cfg=cfg, sd=checkpoint['model'])
            self.epoch = 0
            self.log_buffer = []
            self.t_data, self.t_model = 0, 0
            self.evaluate_epoch()
            self.visualize_epoch()
            self.log(', '.join(self.log_buffer))
        else:
            self.make_model()
            self.train()

        if self.enable_tb:
            self.writer.close()
        if self.enable_wandb:
            wandb.finish()

    def make_datasets(self):
        """
            By default, train dataset performs shuffle and drop_last.
            Distributed sampler will extend the dataset with a prefix to make the length divisible by tot_gpus, samplers should be stored in .dist_samplers.

            Cfg example:

            train/test_dataset:
                name:
                args:
                loader: {batch_size: , num_workers: }
        """
        self.dist_samplers = []

        def make_distributed_loader(dataset, batch_size, num_workers, shuffle=False, drop_last=False):
            sampler = DistributedSampler(dataset, shuffle=shuffle) if self.distributed else None
            loader = DataLoader(
                dataset,
                batch_size // self.total_gpus,
                drop_last=drop_last,
                sampler=sampler,
                shuffle=(shuffle and (sampler is None)),
                num_workers=num_workers // self.total_gpus,
                pin_memory=True)
            return loader, sampler

        if self.train_ds is not None:
            self.train_loader, train_sampler = make_distributed_loader(
                self.train_ds, self.cfg.trainer.batch_size(), self.cfg.trainer.n_workers(), shuffle=True, drop_last=True)
            self.dist_samplers.append(train_sampler)
        
        if self.test_ds is not None:
            self.test_loader, test_sampler = make_distributed_loader(
                self.test_ds, self.cfg.trainer.batch_size(), self.cfg.trainer.n_workers(), shuffle=False, drop_last=False)
            self.dist_samplers.append(test_sampler)

    def make_model(self, cfg=None, sd=None):
        if cfg is None:
            cfg = self.cfg
        model = models.make(model_name=cfg.hypernet.model(), cfg=cfg, sd=sd)
        self.log(f'Model: #params={utils.compute_num_params(model)}, #trainable-params={utils.compute_num_params(model, trainable_only=True)}')

        if self.distributed:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model.cuda()
            model_ddp = DistributedDataParallel(model, device_ids=[self.rank])
        else:
            model.cuda()
            model_ddp = model
        self.model = model
        self.model_ddp = model_ddp

    def train(self):
        """
            For epochs perform training, evaluation, and visualization.
            Note that ave_scalars update ignores the actual current batch_size.
        """
        cfg = self.cfg

        self.optimizer = utils.make_optimizer(self.model_ddp.parameters(), cfg)

        max_epoch = cfg.trainer.max_epoch()
        eval_epoch = cfg.trainer.eval_epoch()
        vis_epoch = cfg.trainer.vis_epoch()
        save_epoch = cfg.trainer.save_epoch()
        epoch_timer = utils.EpochTimer(max_epoch)

        for epoch in range(1, max_epoch + 1):
            self.epoch = epoch
            self.log_buffer = [f'Epoch {epoch}']

            if self.distributed:
                for sampler in self.dist_samplers:
                    sampler.set_epoch(epoch)

            self.adjust_learning_rate()

            self.t_data, self.t_model = 0, 0
            self.train_epoch()

            if epoch % eval_epoch == 0:
                self.evaluate_epoch()
            
            if epoch % vis_epoch == 0:
                self.visualize_epoch()

            if epoch % save_epoch == 0:
                self.save_checkpoint(f'epoch-{epoch}.pth')
            self.save_checkpoint('epoch-last.pth')

            epoch_time, tot_time, est_time = epoch_timer.epoch_done()
            t_data_ratio = self.t_data / (self.t_data + self.t_model)
            self.log_buffer.append(f'{epoch_time} (d {t_data_ratio:.2f}) {tot_time}/{est_time}')
            self.log(', '.join(self.log_buffer))

    def adjust_learning_rate(self):
        base_lr = self.cfg.trainer.optimizer.args.lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = base_lr
        self.log_temp_scalar('lr', self.optimizer.param_groups[0]['lr'])

    def log_temp_scalar(self, k, v, t=None):
        if t is None:
            t = self.epoch
        if self.enable_tb:
            self.writer.add_scalar(k, v, global_step=t)
        if self.enable_wandb:
            wandb.log({k: v}, step=t)

    def dist_all_reduce_mean_(self, x):
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        x.div_(self.total_gpus)

    def sync_ave_scalars_(self, ave_scalars):
        for k in ave_scalars.keys():
            x = torch.tensor(ave_scalars[k].item(), dtype=torch.float32, device=self.device)
            self.dist_all_reduce_mean_(x)
            ave_scalars[k].v = x.item()
            ave_scalars[k].n *= self.total_gpus

    @abstractmethod
    def train_step(self, data):
        loss = None
        return {'loss': loss}

    def train_epoch(self):
        self.model_ddp.train()
        ave_scalars = dict()

        pbar = self.train_loader
        if self.is_master:
            pbar = tqdm(pbar, desc='train', leave=False)

        t1 = time.time()
        for data in pbar:
            t0 = time.time()
            self.t_data += t0 - t1
            ret = self.train_step(data)
            self.t_model += time.time() - t0

            B = len(next(iter(data.values())))
            for k, v in ret.items():
                if ave_scalars.get(k) is None:
                    ave_scalars[k] = utils.Averager()
                ave_scalars[k].add(v, n=B)

            if self.is_master:
                pbar.set_description(desc=f'train: loss={ret["loss"]:.4f}') # type: ignore
            t1 = time.time()

        if self.distributed:
            self.sync_ave_scalars_(ave_scalars)

        logtext = 'train:'
        for k, v in ave_scalars.items():
            logtext += f' {k}={v.item():.4f}'
            self.log_temp_scalar('train/' + k, v.item())
        self.log_buffer.append(logtext)

    @abstractmethod
    def evaluate_step(self, data):
        with torch.no_grad():
            loss = None
        return {'loss': loss}

    def evaluate_epoch(self):
        self.model_ddp.eval()
        ave_scalars = dict()

        pbar = self.test_loader
        if self.is_master:
            pbar = tqdm(pbar, desc='eval', leave=False)

        t1 = time.time()
        for data in pbar:
            t0 = time.time()
            self.t_data += t0 - t1
            ret = self.evaluate_step(data)
            self.t_model += time.time() - t0

            B = len(next(iter(data.values())))
            for k, v in ret.items():
                if ave_scalars.get(k) is None:
                    ave_scalars[k] = utils.Averager()
                ave_scalars[k].add(v, n=B)

            t1 = time.time()

        if self.distributed:
            self.sync_ave_scalars_(ave_scalars)

        logtext = 'eval:'
        for k, v in ave_scalars.items():
            logtext += f' {k}={v.item():.4f}'
            self.log_temp_scalar('test/' + k, v.item())
        self.log_buffer.append(logtext)

    def visualize_epoch(self):
        pass

    def save_checkpoint(self, filename):
        if not self.is_master:
            return
        model_state = self.model.state_dict()
        optimizer_state = self.optimizer.state_dict()
        checkpoint = {
            'model': model_state,
            'optimizer': optimizer_state,
            'epoch': self.epoch,
            'cfg': self.cfg.to_dict(),
        }
        torch.save(checkpoint, osp.join(self.cfg.env.save_dir(), filename))