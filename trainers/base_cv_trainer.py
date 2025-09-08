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
import gc
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

import models
import utils
from trainers import register, BaseTrainer

TRAINER_NAME = "base_cv_trainer"

@register(TRAINER_NAME)
class BaseCVTrainer(BaseTrainer):
    """
    Extension of BaseTrainer for k-fold cross validation.

    - train_ds: list of dicts, each dict = {'train': Dataset, 'val': Dataset}
    - test_ds: ignored
    - Training logs aggregated across folds
    - Validation metrics tracked per fold and averaged at the end
    """

    def make_datasets(self):
        self.folds = []
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
                pin_memory=True
            )
            return loader, sampler

        for i, fold in enumerate(self.train_ds):
            fold_dict = {}

            train_loader, train_sampler = make_distributed_loader(
                fold['train'],
                self.cfg.trainer.batch_size(),
                self.cfg.trainer.n_workers(),
                shuffle=True,
                drop_last=True
            )
            self.dist_samplers.append(train_sampler)
            self.log(f"[Fold {i}] Train dataset size: {len(fold['train'])}")
            fold_dict['train_loader'] = train_loader

            val_loader, val_sampler = make_distributed_loader(
                fold['val'],
                self.cfg.trainer.batch_size(),
                self.cfg.trainer.n_workers(),
                shuffle=False,
                drop_last=False
            )
            self.dist_samplers.append(val_sampler)
            self.log(f"[Fold {i}] Val dataset size: {len(fold['val'])}")
            fold_dict['val_loader'] = val_loader

            self.folds.append(fold_dict)

    def run(self):
        self.make_datasets()
        all_fold_metrics = []
        wandb.define_metric("fold_*", step_metric="fold_step")

        for i, fold in enumerate(self.folds):
            self.log(f"===== Starting Fold {i+1}/{len(self.folds)} =====")

            # fresh model + optimizer per fold
            self.make_model()
            self.optimizer = utils.make_optimizer(self.model_ddp.parameters(), self.cfg)

            # attach loaders for this fold
            self.train_loader = fold['train_loader']
            self.val_loader = fold['val_loader']
            self.current_fold = i

            self.epoch = 0
            self.t_data, self.t_model = 0, 0

            self.train()

            # collect last eval results
            last_metrics = getattr(self, "last_eval_metrics", None)
            if last_metrics:
                all_fold_metrics.append(last_metrics)
            
            del self.model, self.model_ddp, self.optimizer
            torch.cuda.empty_cache()
            gc.collect()

        # --- aggregate CV results ---
        import math
        avg_metrics, std_metrics = {}, {}
        for k in all_fold_metrics[0].keys():
            vals = [m[k] for m in all_fold_metrics]
            mean = sum(vals) / len(vals)
            var = sum((x - mean) ** 2 for x in vals) / len(vals)
            std = math.sqrt(var)
            avg_metrics[k] = mean
            std_metrics[k] = std

        self.log("===== Cross Validation Results =====")
        for k in avg_metrics.keys():
            self.log(f"{k}: mean={avg_metrics[k]:.4f}, std={std_metrics[k]:.4f}")
            self.log_temp_scalar(f"cv/{k}_mean", avg_metrics[k], t=0)
            self.log_temp_scalar(f"cv/{k}_std", std_metrics[k], t=0)


    def train(self):
        """
            Exactly the same as BaseTrainer.train(), but with saving fold-specific checkpoints
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
                self.save_checkpoint(f'epoch-fold{self.current_fold}-{epoch}.pth')
            self.save_checkpoint(f'epoch-fold{self.current_fold}-last.pth')

            epoch_time, tot_time, est_time = epoch_timer.epoch_done()
            t_data_ratio = self.t_data / (self.t_data + self.t_model)
            self.log_buffer.append(f'{epoch_time} (d {t_data_ratio:.2f}) {tot_time}/{est_time}')
            self.log(', '.join(self.log_buffer))

    def log_temp_scalar(self, k, v, t=None):
        if t is None:
            t = self.epoch
        if self.enable_tb:
            self.writer.add_scalar(k, v, global_step=t)
        if self.enable_wandb:
            wandb.log({f'fold_{self.current_fold}/{k}': v, 'fold_step': self.epoch})

    def adjust_learning_rate(self):
        base_lr = self.cfg.trainer.optimizer.args.lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = base_lr
        self.log_temp_scalar(f'lr', self.optimizer.param_groups[0]['lr'])

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
            self.log_temp_scalar(f'train/' + k, v.item())
        self.log_buffer.append(logtext)

    def evaluate_epoch(self):
        self.model_ddp.eval()
        ave_scalars = dict()

        pbar = self.val_loader
        if self.is_master:
            pbar = tqdm(pbar, desc=f'eval (fold {self.current_fold})', leave=False)

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

        logtext = f'eval (fold {self.current_fold}):'
        metrics_dict = {}
        for k, v in ave_scalars.items():
            logtext += f' {k}={v.item():.4f}'
            # log fold-specific metrics
            self.log_temp_scalar(f'eval/' + k, v.item())
            metrics_dict[k] = v.item()
        self.log_buffer.append(logtext)

        self.last_eval_metrics = metrics_dict
