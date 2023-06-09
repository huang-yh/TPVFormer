_base_ = [
    './_base_/dataset.py',
    './_base_/optimizer.py',
    './_base_/schedule.py',
]
model_type = '10'
occ_path = 'data/nuscenes-occupancy'
load_from = ''
scale_rate = 0.5
max_epochs = 15

dataset_params = dict(
    version = "v1.0-trainval",
    ignore_label = 17,
    fill_label = 0,
    fixed_volume_space = True,
    label_mapping = "./config/label_mapping/nuscenes-openOccupancy.yaml",
    max_volume_space = [51.2, 51.2, 3],
    min_volume_space = [-51.2, -51.2, -5],
)

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

_dim_ = 64
num_heads = 4
_pos_dim_ = [24, 24, 16]
_ffn_dim_ = _dim_*2
_num_levels_ = 4
_num_cams_ = 6

tpv_h_ = 256
tpv_w_ = 256
tpv_z_ = 20
scale_h = 2
scale_w = 2
scale_z = 2
tpv_encoder_layers = 5
num_points_in_pillar = [4, 32, 32]
num_points = [4, 32, 32]
hybrid_attn_anchors = 16
hybrid_attn_points = 16
hybrid_attn_init = 0

grid_size = [tpv_h_*scale_h, tpv_w_*scale_w, tpv_z_*scale_z]
nbr_class = 18

self_cross_layer = dict(
    type='TPVFormerLayer',
    attn_cfgs=[
        dict(
            type='TPVCrossViewHybridAttention',
            tpv_h=tpv_h_,
            tpv_w=tpv_w_,
            tpv_z=tpv_z_,
            num_anchors=hybrid_attn_anchors,
            embed_dims=_dim_,
            num_heads=num_heads,
            num_points=hybrid_attn_points,
            init_mode=hybrid_attn_init,
        ),
        dict(
            type='TPVImageCrossAttention',
            pc_range=point_cloud_range,
            num_cams=_num_cams_,
            deformable_attention=dict(
                type='TPVMSDeformableAttention3D',
                embed_dims=_dim_,
                num_heads=num_heads,
                num_points=num_points,
                num_z_anchors=num_points_in_pillar,
                num_levels=_num_levels_,
                floor_sampling_offset=False,
                tpv_h=tpv_h_,
                tpv_w=tpv_w_,
                tpv_z=tpv_z_,
            ),
            embed_dims=_dim_,
            tpv_h=tpv_h_,
            tpv_w=tpv_w_,
            tpv_z=tpv_z_,
        )
    ],
    feedforward_channels=_ffn_dim_,
    ffn_dropout=0.1,
    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
)

self_layer = dict(
    type='TPVFormerLayer',
    attn_cfgs=[
        dict(
            type='TPVCrossViewHybridAttention',
            tpv_h=tpv_h_,
            tpv_w=tpv_w_,
            tpv_z=tpv_z_,
            num_anchors=hybrid_attn_anchors,
            embed_dims=_dim_,
            num_heads=num_heads,
            num_points=hybrid_attn_points,
            init_mode=hybrid_attn_init,
        )
    ],
    feedforward_channels=_ffn_dim_,
    ffn_dropout=0.1,
    operation_order=('self_attn', 'norm', 'ffn', 'norm')
)


model = dict(
    type='TPVFormer',
    use_grid_mask=True,
    tpv_aggregator=dict(
        type='TPVOccAggregator',
        tpv_h=tpv_h_,
        tpv_w=tpv_w_,
        tpv_z=tpv_z_,
        nbr_classes=nbr_class,
        in_dims=_dim_,
        hidden_dims=2*_dim_,
        out_dims=_dim_,
        scale_h=scale_h,
        scale_w=scale_w,
        scale_z=scale_z
    ),
    # img_backbone=dict(
    #     type='ResNet',
    #     depth=101,
    #     num_stages=4,
    #     out_indices=(1, 2, 3),
    #     frozen_stages=1,
    #     norm_cfg=dict(type='BN2d', requires_grad=False),
    #     norm_eval=True,
    #     style='caffe',
    #     dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False), # original DCNv2 will print log when perform load_state_dict
    #     stage_with_dcn=(False, False, True, True)),
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=0,
        norm_eval=False,
        style='pytorch',
        pretrained='./ckpts/resnet50-0676ba61.pth'),
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True),
    tpv_head=dict(
        type='TPVFormerHead',
        tpv_h=tpv_h_,
        tpv_w=tpv_w_,
        tpv_z=tpv_z_,
        pc_range=point_cloud_range,
        num_feature_levels=_num_levels_,
        num_cams=_num_cams_,
        embed_dims=_dim_,
        encoder=dict(
            type='TPVFormerEncoder',
            tpv_h=tpv_h_,
            tpv_w=tpv_w_,
            tpv_z=tpv_z_,
            num_layers=tpv_encoder_layers,
            pc_range=point_cloud_range,
            num_points_in_pillar=num_points_in_pillar,
            num_points_in_pillar_cross_view=[16, 16, 16],
            return_intermediate=False,
            transformerlayers=[
                self_cross_layer,
                self_cross_layer,
                self_cross_layer,
                self_layer,
                self_layer,
            ]),
        positional_encoding=dict(
            type='CustomPositionalEncoding',
            num_feats=_pos_dim_,
            h=tpv_h_,
            w=tpv_w_,
            z=tpv_z_
        )))