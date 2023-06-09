
import torch, torch.nn as nn, torch.nn.functional as F
from mmcv.runner import BaseModule
from mmseg.models import HEADS


@HEADS.register_module()
class TPVOccAggregator(BaseModule):
    def __init__(
        self, tpv_h, tpv_w, tpv_z, nbr_classes=20, 
        in_dims=64, hidden_dims=128, out_dims=None,
        scale_h=2, scale_w=2, scale_z=2, 
        use_checkpoint=False, return_vox=False,
        return_pts=True, pc_origin=[-51.2, -51.2, -5],
        pc_range=[102.4, 102.4, 8.]
    ):
        super().__init__()
        self.tpv_h = tpv_h
        self.tpv_w = tpv_w
        self.tpv_z = tpv_z
        self.scale_h = scale_h
        self.scale_w = scale_w
        self.scale_z = scale_z

        out_dims = in_dims if out_dims is None else out_dims

        self.decoder = nn.Sequential(
            nn.Conv1d(in_dims, hidden_dims, 1),
            nn.Softplus(),
            nn.Conv1d(hidden_dims, out_dims, 1)
        )

        self.classifier = nn.Conv1d(out_dims, nbr_classes, 1)
        self.classes = nbr_classes
        self.use_checkpoint = use_checkpoint
        self.return_vox = return_vox
        self.return_pts = return_pts
        self.batch_size = 256 * 256 * 20 * 4
        self.register_buffer('pc_origin', torch.tensor(pc_origin)[None, None, ...], False)
        self.register_buffer('pc_range', torch.tensor(pc_range)[None, None, ...], False)
    
    def forward(self, tpv_list, points=None):
        """
        tpv_list[0]: bs, h*w, c
        tpv_list[1]: bs, z*h, c
        tpv_list[2]: bs, w*z, c
        """
        if self.return_pts:
            assert points is not None
        
        tpv_hw, tpv_zh, tpv_wz = tpv_list[0], tpv_list[1], tpv_list[2]
        bs, _, c = tpv_hw.shape
        tpv_hw = tpv_hw.permute(0, 2, 1).reshape(bs, c, self.tpv_h, self.tpv_w)
        tpv_zh = tpv_zh.permute(0, 2, 1).reshape(bs, c, self.tpv_z, self.tpv_h)
        tpv_wz = tpv_wz.permute(0, 2, 1).reshape(bs, c, self.tpv_w, self.tpv_z)

        if self.return_pts:
            # points: bs, n, 3
            _, n, _ = points.shape
            points = (points - self.pc_origin) / self.pc_range * 2 - 1
            points = points.reshape(bs, 1, n, 3)
            sample_loc = points[:, :, :, [0, 1]]
            tpv_hw_pts = F.grid_sample(tpv_hw, sample_loc).squeeze(2) # bs, c, n
            sample_loc = points[:, :, :, [1, 2]]
            tpv_zh_pts = F.grid_sample(tpv_zh, sample_loc).squeeze(2)
            sample_loc = points[:, :, :, [2, 0]]
            tpv_wz_pts = F.grid_sample(tpv_wz, sample_loc).squeeze(2)
        
            fused_pts = tpv_hw_pts + tpv_zh_pts + tpv_wz_pts
            fused_pts = self.decoder(fused_pts)
            logits_pts = self.classifier(fused_pts)
            logits_pts = logits_pts.reshape(bs, self.classes, n, 1, 1)
            return logits_pts

        if self.return_vox:
            if self.scale_h != 1 or self.scale_w != 1:
                tpv_hw = F.interpolate(
                    tpv_hw, 
                    size=(self.tpv_h*self.scale_h, self.tpv_w*self.scale_w),
                    mode='bilinear'
                )
            if self.scale_z != 1 or self.scale_h != 1:
                tpv_zh = F.interpolate(
                    tpv_zh, 
                    size=(self.tpv_z*self.scale_z, self.tpv_h*self.scale_h),
                    mode='bilinear'
                )
            if self.scale_w != 1 or self.scale_z != 1:
                tpv_wz = F.interpolate(
                    tpv_wz, 
                    size=(self.tpv_w*self.scale_w, self.tpv_z*self.scale_z),
                    mode='bilinear'
                )
        
            tpv_hw = tpv_hw.unsqueeze(-1).permute(0, 1, 3, 2, 4).expand(-1, -1, -1, -1, self.scale_z*self.tpv_z)
            tpv_zh = tpv_zh.unsqueeze(-1).permute(0, 1, 4, 3, 2).expand(-1, -1, self.scale_w*self.tpv_w, -1, -1)
            tpv_wz = tpv_wz.unsqueeze(-1).permute(0, 1, 2, 4, 3).expand(-1, -1, -1, self.scale_h*self.tpv_h, -1)
            fused = (tpv_hw + tpv_zh + tpv_wz).flatten(2)

            output = []
            cnt = 0
            while cnt < fused.shape[-1]:
                end = min(cnt + self.batch_size, fused.shape[-1])
                batch = fused[..., cnt:end]
                batch = self.classifier(self.decoder(batch))
                output.append(batch)
                cnt += self.batch_size
            logits = torch.cat(output, dim=-1)
            
            # fused = self.decoder(fused)
            # logits = self.classifier(fused)
            logits = logits.reshape(bs, self.classes, self.scale_w*self.tpv_w, self.scale_h*self.tpv_h, self.scale_z*self.tpv_z)
        
            return logits
        
        raise NotImplementedError
