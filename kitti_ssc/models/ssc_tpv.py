import torch
import torch.nn as nn
from ..loss.sscMetrics import SSCMetrics
from ..loss.ssc_loss import sem_scal_loss, CE_ssc_loss, KL_sep, geo_scal_loss
import numpy as np
import torch.nn.functional as F
from .unet2d import UNet2D

from builder import tpv10_builder
from torch.utils.tensorboard import SummaryWriter

class SSCTPV(nn.Module):
    def __init__(
        self,
        model_cfg,
        n_classes,
        class_names,
        feature,
        class_weights,
        fp_loss=True,
        frustum_size=4,
        CE_ssc_loss=True,
        geo_scal_loss=True,
        sem_scal_loss=True,
        tflogger=None,
        **kwargs,
    ):
        super().__init__()

        self.fp_loss = fp_loss
        self.frustum_size = frustum_size
        self.class_names = class_names
        self.CE_ssc_loss = CE_ssc_loss
        self.sem_scal_loss = sem_scal_loss
        self.geo_scal_loss = geo_scal_loss
        self.class_weights = class_weights

        self.n_classes = n_classes
        self.net_rgb = UNet2D.build(
            out_feature=feature, 
            use_decoder=True)
        self.net_3d_decoder = tpv10_builder.build(model_cfg)
        
        self.train_metrics = SSCMetrics(self.n_classes)
        self.val_metrics = SSCMetrics(self.n_classes)
        self.tflogger = tflogger

    def log(self, tag, tensor, global_step):
        if self.tflogger is None:
            return 
        assert isinstance(self.tflogger, SummaryWriter)
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.item()
        self.tflogger.add_scalar(tag, tensor, global_step)

    def forward_dummy(self, batch):
        img = batch["img"]
        img_metas = batch["img_metas"]
        res = {}

        img = batch['x_rgb']
        out = self.net_3d_decoder(img=img, img_metas=img_metas)
        res["ssc_logit"] = out

        return res

    def forward_model(self, batch):

        img = batch["img"]
        out = {}
        x_rgb = self.net_rgb(img)

        input_dict = {
            "img": [
                x_rgb[0].unsqueeze(1), # B, N, C, H, W
                x_rgb[1].unsqueeze(1),
                x_rgb[2].unsqueeze(1),
                x_rgb[3].unsqueeze(1)
            ],
            "img_metas": batch["img_metas"]
        }

        out = self.net_3d_decoder(**input_dict)
        out = out.permute(0, 1, 3, 2, 4)
        out = out[:, :, :, list(reversed(range(out.shape[3]))), :]
        return {'ssc_logit': out}

    def forward(self, batch, step_type, global_iter):
        if step_type == 'train':
            metric = self.train_metrics
        elif step_type == 'val':
            metric = self.val_metrics

        bs = len(batch["img"])
        loss = 0
        out_dict = self.forward_model(batch)
        ssc_pred = out_dict["ssc_logit"]
        target = batch["target"]

        class_weight = self.class_weights.type_as(batch["img"])
        if self.CE_ssc_loss:
            loss_ssc = CE_ssc_loss(ssc_pred, target, class_weight)
            loss += loss_ssc
            self.log(
                step_type + "/loss_ssc",
                loss_ssc.detach(),
                global_iter)

        if self.sem_scal_loss:
            loss_sem_scal = sem_scal_loss(ssc_pred, target)
            loss += loss_sem_scal
            self.log(
                step_type + "/loss_sem_scal",
                loss_sem_scal.detach(),
                global_iter)

        if self.geo_scal_loss:
            loss_geo_scal = geo_scal_loss(ssc_pred, target)
            loss += loss_geo_scal
            self.log(
                step_type + "/loss_geo_scal",
                loss_geo_scal.detach(),
                global_iter)

        if self.fp_loss and step_type != "test":
            frustums_masks = batch["frustums_masks"]
            frustums_class_dists = batch["frustums_class_dists"]  # (bs, n_frustums, n_classes)
            n_frustums = frustums_class_dists.shape[1]

            pred_prob = F.softmax(ssc_pred, dim=1)
            batch_cnt = frustums_class_dists.sum(0)  # (n_frustums, n_classes)

            frustum_loss = 0
            frustum_nonempty = 0
            for frus in range(n_frustums):
                frustum_mask = frustums_masks[:, frus, :, :, :].unsqueeze(1).float()
                prob = frustum_mask * pred_prob  # bs, n_classes, H, W, D
                prob = prob.reshape(bs, self.n_classes, -1).permute(1, 0, 2)
                prob = prob.reshape(self.n_classes, -1)
                cum_prob = prob.sum(dim=1)  # n_classes

                total_cnt = torch.sum(batch_cnt[frus])
                total_prob = prob.sum()
                if total_prob > 0 and total_cnt > 0:
                    frustum_target_proportion = batch_cnt[frus] / total_cnt
                    cum_prob = cum_prob / total_prob  # n_classes
                    frustum_loss_i = KL_sep(cum_prob, frustum_target_proportion)
                    frustum_loss += frustum_loss_i
                    frustum_nonempty += 1
            frustum_loss = frustum_loss / frustum_nonempty
            loss += frustum_loss
            self.log(
                step_type + "/loss_frustums",
                frustum_loss.detach(),
                global_iter)

        y_true = target.cpu().numpy()
        y_pred = ssc_pred.detach().cpu().numpy()
        y_pred = np.argmax(y_pred, axis=1)
        metric.add_batch(y_pred, y_true)

        self.log(step_type + "/loss", loss.detach(), global_iter)
        return loss

    def validation_epoch_end(self, epoch):
        metric_list = [("train", self.train_metrics), ("val", self.val_metrics)]
        miou = dict()

        for prefix, metric in metric_list:
            stats = metric.get_stats()
            for i, class_name in enumerate(self.class_names):
                self.log(
                    "{}_SemIoU/{}".format(prefix, class_name),
                    stats["iou_ssc"][i],
                    epoch)
            miou[prefix] = stats["iou_ssc_mean"]
            self.log("{}/mIoU".format(prefix), stats["iou_ssc_mean"], epoch)
            self.log("{}/IoU".format(prefix), stats["iou"], epoch)
            self.log("{}/Precision".format(prefix), stats["precision"], epoch)
            self.log("{}/Recall".format(prefix), stats["recall"], epoch)
            metric.reset()
        return miou