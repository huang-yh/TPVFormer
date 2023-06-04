from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from .kitti_dataset import KittiDataset
from .collate import collate_fn
from ..utils.torch_util import worker_init_fn
    
def get_dataloader(
    root,
    preprocess_root,
    frustum_size=4,
    batch_size=4,
    num_workers=6,
    dist=False):

    train_ds = KittiDataset(
        split="train",
        root=root,
        preprocess_root=preprocess_root,
        frustum_size=frustum_size,
        fliplr=0.5,
        color_jitter=(0.4, 0.4, 0.4),
    )

    val_ds = KittiDataset(
        split="val",
        root=root,
        preprocess_root=preprocess_root,
        frustum_size=frustum_size,
        fliplr=0,
        color_jitter=None,
    )

    # do not use ddp in inference, to keep metrics consistent among gpus
    if dist:
        train_sampler = DistributedSampler(train_ds, drop_last=True)
        val_sampler = DistributedSampler(val_ds, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True if train_sampler is None else False,
        sampler=train_sampler,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_fn,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )
    return train_dl, val_dl