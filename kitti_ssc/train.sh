
export KITTI_PREPROCESS=/home/wenzhao/datasets/SemanticKITTI/preprocess
export KITTI_ROOT=/home/wenzhao/datasets/SemanticKITTI
export KITTI_LOG=/home/wenzhao/hyh/MonoScene/logs/kitti

python kitti_ssc/scripts/train.py \
    dataset=kitti \
    enable_log=true \
    kitti_root=$KITTI_ROOT \
    kitti_preprocess_root=$KITTI_PREPROCESS\
    kitti_logdir=$KITTI_LOG \
    model_cfg=kitti_ssc/configs/tpv10_kitti_ssc.py