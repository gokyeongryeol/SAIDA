_base_ = ['../external/mmdetection/projects/CO-DETR/configs/codino/co_dino_5scale_swin_l_16xb1_16e_o365tococo.py']

import os
import subprocess
data_root = os.environ.get("DATA_ROOT", "/dataset/")

import json
categories = json.load(open(os.path.join(data_root, 'annotations/1_shot.json')))["categories"]
num_classes = len(categories)
metainfo = {
    'classes': tuple([cat_dict['name'] for cat_dict in categories]),
}

checkpoint = os.environ.get("CHECKPOINT")
if checkpoint:
    load_from = checkpoint

train_pipeline = _base_.train_pipeline

max_epochs = int(os.environ.get("MAX_EPOCHS", "16"))
base_lr = float(os.environ.get("BASE_LR", "1e-4"))
train_batch_size = int(os.environ.get("TRAIN_BATCH_SIZE", "1"))
val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", "1"))
val_interval = int(os.environ.get("VAL_INTERVAL", "1"))
num_gpu = int(os.environ.get("GPUS", "1"))
num_syn = int(os.environ.get("SYNS", "0"))

use_color_aug = os.environ.get("USE_COLOR_AUG", "false").lower() == "true"
if use_color_aug:
    color_space = [
        [dict(type='ColorTransform')],
        [dict(type='AutoContrast')],
        [dict(type='Equalize')],
        [dict(type='Sharpness')],
        [dict(type='Posterize')],
        [dict(type='Solarize')],
        [dict(type='Color')],
        [dict(type='Contrast')],
        [dict(type='Brightness')],
    ]

    train_pipeline.insert(-1, dict(
        type='RandomOrder',
        transforms=[
            dict(type='RandAugment', aug_space=color_space, aug_num=1),
        ]
    ))

# model
num_dec_layer = 6
loss_lambda = 2.0
model = dict(
    query_head=dict(num_classes=num_classes),
    roi_head=[
        dict(
            type='CoStandardRoIHead',
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32, 64],
                finest_scale=56),
            bbox_head=dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0 * num_dec_layer * loss_lambda),
                loss_bbox=dict(
                    type='GIoULoss',
                    loss_weight=10.0 * num_dec_layer * loss_lambda)))
    ],
    bbox_head=[
        dict(
            type='CoATSSHead',
            num_classes=num_classes,
            in_channels=256,
            stacked_convs=1,
            feat_channels=256,
            anchor_generator=dict(
                type='AnchorGenerator',
                ratios=[1.0],
                octave_base_scale=8,
                scales_per_octave=1,
                strides=[4, 8, 16, 32, 64, 128]),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[.0, .0, .0, .0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0 * num_dec_layer * loss_lambda),
            loss_bbox=dict(
                type='GIoULoss',
                loss_weight=2.0 * num_dec_layer * loss_lambda),
            loss_centerness=dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                loss_weight=1.0 * num_dec_layer * loss_lambda)),
    ],
)


log_processor = dict(
    by_epoch=True,
    type='LogProcessor',
    window_size=val_interval)

train_cfg = dict(
    max_epochs=max_epochs,
    type='EpochBasedTrainLoop',
    val_interval=val_interval)


train_dataloader = dict(
    batch_size=train_batch_size,
    pin_memory=False,
    dataset=dict(
        _delete_=True,
        type='ConcatDataset',
        datasets=[
            dict(
                type='CocoDataset',
                data_root=data_root,
                metainfo=metainfo,
                ann_file='annotations/1_shot.json',
                data_prefix=dict(img='train/'),
                pipeline=train_pipeline,
                filter_cfg=dict(filter_empty_gt=False, min_size=32),
            )
        ] + [
            dict(
                type='CocoDataset',
                data_root=data_root,
                metainfo=metainfo,
                ann_file=f'annotations/syn_t2i_{j}.json',
                data_prefix=dict(img=f'syn_t2i_{j}/'),
                pipeline=train_pipeline,
                filter_cfg=dict(filter_empty_gt=False, min_size=32),
            ) for j in range(num_syn)
        ]
    )
)


val_dataloader = dict(
    batch_size=val_batch_size,
    pin_memory=False,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/test.json',
        data_prefix=dict(img='test/')))

test_dataloader = val_dataloader

# metric
val_evaluator = dict(
    metric=['bbox'],
    ann_file=os.path.join(data_root, 'annotations/test.json'), classwise=True)

test_evaluator = val_evaluator

# save hook
default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        save_best="coco/bbox_mAP",
        rule="greater",
        max_keep_ckpts=1,
    )
)

# optimizer
optim_wrapper = dict(
    optimizer=dict(lr=base_lr),
    accumulative_counts=max(8 // num_gpu, 1),
)

param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        end=max_epochs,
        gamma=0.1,
        milestones=[
            max_epochs // 2,
        ],
        type='MultiStepLR'),
]

# tta
tta_model = dict(
    type='DetTTAModel',
    tta_cfg=dict(
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100,
    ),
)

tta_img_scales = [(1536, 960), (2048, 1280), (2560, 1600)]

tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale=scale, keep_ratio=True)
                for scale in tta_img_scales
            ],
            [
                dict(type='RandomFlip', prob=1.),
                dict(type='RandomFlip', prob=0.)
            ],
            [dict(type='LoadAnnotations', with_bbox=True)],
            [
                dict(
                    type='PackDetInputs',
                    meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                               'scale_factor', 'flip', 'flip_direction'))
            ]
        ])
]