import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(__file__), os.pardir, os.pardir)))

from kitti_ssc.dataset.semantic_kitti.kitti_dm import get_dataloader
from kitti_ssc.dataset.semantic_kitti.params import (
    semantic_kitti_class_frequencies,
    kitti_class_names,
)
import argparse
from torch.utils.tensorboard import SummaryWriter
import os.path as osp, time
import numpy as np
import torch, torch.nn as nn
import torch.distributed as dist
import torch.nn.parallel as parallel
from timm.scheduler import CosineLRScheduler
import mmcv
from mmcv import Config
from mmcv.utils import get_logger

def main(local_rank=0, args=None):
    if local_rank != 0:
        import builtins
        def pass_print(*args, **kwargs):
            pass
        builtins.print = pass_print

    config = Config.fromfile(args.py_config)
    
    config.lr = args.gpus / 8 * config.lr
    ## prepare exp_name
    exp_name = config.exp_prefix
    exp_name += "_FrusSize_{}".format(config.frustum_size)
    exp_name += "_WD{}_lr{}".format(config.weight_decay, config.lr)
    if config.CE_ssc_loss:
        exp_name += "_CEssc"
    if config.geo_scal_loss:
        exp_name += "_geoScalLoss"
    if config.sem_scal_loss:
        exp_name += "_semScalLoss"
    if config.fp_loss:
        exp_name += "_fpLoss"
    print(exp_name)
    # args.work_dir = os.path.join(args.work_dir, exp_name)

    # prepare ddp
    if args.gpus > 1:
        distributed = True
    else:
        distributed = False
    if distributed:
        ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = os.environ.get("MASTER_PORT", "20506")
        hosts = int(os.environ.get("WORLD_SIZE", 1))  # number of nodes
        rank = int(os.environ.get("RANK", 0))  # node id
        gpus = torch.cuda.device_count()  # gpus per node
        print(f"tcp://{ip}:{port}")
        dist.init_process_group(
            backend="nccl", init_method=f"tcp://{ip}:{port}", 
            world_size=hosts * gpus, rank=rank * gpus + local_rank
        )
        world_size = dist.get_world_size()
        config.gpu_ids = range(world_size)
        torch.cuda.set_device(local_rank)

    # configure logger
    if local_rank == 0:
        os.makedirs(args.work_dir, exist_ok=True)
        config.dump(osp.join(args.work_dir, osp.basename(args.py_config)))
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.work_dir, f'{timestamp}.log')
    txtlogger = get_logger(name='ssc', log_file=log_file, log_level='INFO')
    txtlogger.info(f'Config:\n{config.pretty_text}')

    if config.enable_log and local_rank == 0:
        tflogger = SummaryWriter(log_dir=osp.join(args.work_dir, 'tf'))
    else:
        tflogger = None

    # Setup dataloaders
    max_epochs = config.max_epochs
    train_loader, val_loader = get_dataloader(
        root=args.kitti_root,
        preprocess_root=args.kitti_preprocess_root,
        frustum_size=config.frustum_size,
        batch_size=config.num_samples_per_gpu,
        num_workers=config.num_workers_per_gpu,
        dist=distributed)

    # Initialize ssc model
    from kitti_ssc.models.ssc_tpv import SSCTPV
    class_names = kitti_class_names
    feature = config.feature
    n_classes = config.nbr_class
    class_weights = torch.from_numpy(
        1 / np.log(semantic_kitti_class_frequencies + 0.001))
    model = SSCTPV(
        model_cfg=config.model,
        frustum_size=config.frustum_size,
        fp_loss=config.fp_loss,
        feature=feature,
        n_classes=n_classes,
        class_names=class_names,
        CE_ssc_loss=config.CE_ssc_loss,
        sem_scal_loss=config.sem_scal_loss,
        geo_scal_loss=config.geo_scal_loss,
        class_weights=class_weights,
        tflogger=tflogger).cuda()
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    txtlogger.info(f'Number of params: {n_parameters}')

    if distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = parallel.DistributedDataParallel(
            model,
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=False)
        raw_model = model.module
    else:
        raw_model = model
    
    # initialize optim and scheduler
    optimizer = torch.optim.AdamW(
        raw_model.parameters(), 
        lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CosineLRScheduler(
        optimizer,
        max_epochs * len(train_loader),
        lr_min=1e-6,
        warmup_t=500,
        warmup_lr_init=1e-6,
        t_in_epochs=False)

    # deal with resume
    config.resume_from = ''
    if osp.exists(osp.join(args.work_dir, 'latest.pth')):
        config.resume_from = osp.join(args.work_dir, 'latest.pth')
    if args.resume_from:
        config.resume_from = args.resume_from    
    print('resume from: ', config.resume_from)

    epoch, global_iter_train, global_iter_val = 0, 0, 0
    if config.resume_from and osp.exists(config.resume_from):
        map_location = 'cpu'
        ckpt = torch.load(config.resume_from, map_location=map_location)
        print(raw_model.load_state_dict(ckpt['state_dict'], strict=True))
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        epoch = ckpt['epoch']
        global_iter_train = ckpt['global_iter_train']
        global_iter_val = ckpt['global_iter_val']
        print(f'successfully resumed from epoch {epoch}')

    
    # start train
    print_freq = config.print_freq
    while epoch < max_epochs:
        model.train()
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        loss_list = []
        time.sleep(10)
        data_time_s = time.time()
        time_s = time.time()
        for i_iter, batch in enumerate(train_loader):
            to_cuda_list = ['img', 'target', 'frustums_class_dists', 'frustums_masks']
            for k in to_cuda_list:
                batch[k] = batch[k].cuda()
            data_time_e = time.time()

            # forward + backward + optimize
            loss = model(batch, 'train', global_iter_train)

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.grad_max_norm)
            optimizer.step()
            loss_list.append(loss.detach().item())
            scheduler.step_update(global_iter_train)
            time_e = time.time()

            global_iter_train += 1
            if i_iter % print_freq == 0 and local_rank == 0:
                lr = optimizer.param_groups[0]['lr']
                txtlogger.info('[TRAIN] Epoch %d Iter %5d/%d: Loss: %.3f (%.3f), grad_norm: %.1f, lr: %.7f, time: %.3f (%.3f)'%(
                    epoch, i_iter, len(train_loader), 
                    loss.detach().item(), np.mean(loss_list), grad_norm, lr,
                    time_e - time_s, data_time_e - data_time_s
                ))
            data_time_s = time.time()
            time_s = time.time()
                
        # eval
        model.eval()
        val_loss_list = []
        with torch.no_grad():
            for i_iter_val, batch in enumerate(val_loader):
                
                to_cuda_list = ['img', 'target', 'frustums_class_dists', 'frustums_masks']
                for k in to_cuda_list:
                    batch[k] = batch[k].cuda()

                loss = model(batch, 'val', global_iter_val)
                val_loss_list.append(loss.detach().item())
                global_iter_val += 1
                if i_iter_val % print_freq == 0 and local_rank == 0:
                    txtlogger.info('[EVAL] Epoch %d Iter %5d/%d: Loss: %.3f (%.3f)'%(
                        epoch, i_iter_val, len(val_loader), loss.item(), np.mean(val_loss_list)))
        
        miou = raw_model.validation_epoch_end(epoch)
        txtlogger.info('Current train miou is %.3f' % (miou['train'], ))
        txtlogger.info('Current val miou is %.3f' % (miou['val'], ))
        txtlogger.info('Current val loss is %.3f' % (np.mean(val_loss_list)))

        # save checkpoint
        if dist.get_rank() == 0:
            dict_to_save = {
                'state_dict': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch + 1,
                'global_iter_train': global_iter_train,
                'global_iter_val': global_iter_val,
                'epoch_miou': miou['val']
            }
            save_file_name = os.path.join(os.path.abspath(args.work_dir), f'epoch_{epoch+1}.pth')
            torch.save(dict_to_save, save_file_name)
            dst_file = osp.join(args.work_dir, 'latest.pth')
            mmcv.symlink(save_file_name, dst_file)

        epoch += 1

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='kitti_ssc/configs/tpv10_kitti_ssc.py')
    parser.add_argument('--kitti-root', default='data/kitti')
    parser.add_argument('--kitti-preprocess-root', default='data/kitti/preprocess_ssc')
    parser.add_argument('--work-dir', type=str, default='kitti_ssc/out/')
    parser.add_argument('--resume-from', type=str, default='')
    args = parser.parse_args()
    
    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    print(args)

    if ngpus > 1:
        torch.multiprocessing.spawn(main, args=(args,), nprocs=args.gpus)
    else:
        main(0, args)
