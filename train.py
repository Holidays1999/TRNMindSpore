# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Model training"""
import os

import numpy as np
from mindspore import context, load_checkpoint, load_param_into_net, Model, nn, set_seed, Tensor
from mindspore import dtype as mstype
from mindspore import ops
from mindspore.communication.management import get_group_size, get_rank, init
from mindspore.context import ParallelMode
from mindspore.nn.loss.loss import LossBase
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.train.callback import CheckpointConfig, LossMonitor, ModelCheckpoint, TimeMonitor

from model_utils.logging import get_logger
from model_utils.moxing_adapter import config
from model_utils.moxing_adapter import sync_data
from model_utils.util import get_param_groups
from src.bn_inception import BNInception
from src.cell import cast_amp
from src.mox_callback import MoxingCallBack
from src.train_cell import CustomTrainOneStepCell, CustomWithLossCell
from src.trn import RelationModuleMultiScale
from src.tsn import TSN
from src.tsn_dataset import get_dataset_for_training
from src.van import van_base

set_seed(config.seed)


def initialize_backbone(cfg, backbone, checkpoint_path):
    """Initialize the BNInception backbone"""
    print("initialize backbone...")
    if cfg.enable_modelarts:
        model_path = "/cache/model.ckpt"
        sync_data(checkpoint_path, model_path)
        checkpoint_path = model_path
    ckpt_data = load_checkpoint(checkpoint_path)

    # The original BN Inception has 1000 model outputs,
    # but we need config.img_feature_dim of the outputs.
    # So we just take the last fc layer from backbone.
    # ckpt_data[] = backbone.fc.weight
    # ckpt_data['fc.bias'] = backbone.fc.bias
    if 'fc.weight' in ckpt_data.keys():
        ckpt_data.pop('fc.weight')
        ckpt_data.pop('fc.bias')
    for key, value in ckpt_data.copy().items():
        if 'head' in key:
            print(f'==> removing {key} with shape {value.shape}')
            ckpt_data.pop(key)
    not_loaded = load_param_into_net(backbone, ckpt_data)
    if not_loaded:
        print(f'The following parameters are not loaded: {not_loaded}')


def get_step_learning_rate(lr_init, lr_decay, total_epochs, update_lr_epochs, steps_per_epoch):
    """Calculate learning rate values"""
    steps = np.arange(total_epochs * steps_per_epoch)
    epochs = steps // steps_per_epoch
    decay_steps = epochs // update_lr_epochs
    lr_values = lr_init * np.power(lr_decay, decay_steps)
    return lr_values.astype("float32")


def prepare_context(cfg):
    """prepare context"""
    # set context and device
    device_target = cfg.device_target
    device_num = int(os.environ.get("DEVICE_NUM", 1))
    context.set_context(mode=context.GRAPH_MODE, device_target=cfg.device_target)
    if device_target == "Ascend":
        if device_num > 1:
            context.set_context(device_id=int(os.environ["DEVICE_ID"]))
            init(backend_name='hccl')
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            cfg.rank = get_rank()
            cfg.group_size = get_group_size()
        else:
            context.set_context(device_id=cfg.device_id)
            cfg.rank = 0
            cfg.group_size = 1
    elif device_target == "GPU":
        if device_num > 1:
            init(backend_name='nccl')
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            cfg.rank = get_rank()
            cfg.group_size = get_group_size()
        else:
            context.set_context(device_id=cfg.device_id)
            cfg.rank = 0
            cfg.group_size = 1
    else:
        raise ValueError("Unsupported platform.")


def prepare_optimizer(cfg, network, dataset_size):
    """Prepare optimizer"""
    print("prepare optimizer...")

    # lr scheduler
    if cfg.lr_schedule == "step":
        lr = get_step_learning_rate(
            lr_init=cfg.lr,
            lr_decay=0.1,
            total_epochs=cfg.epochs_num,
            update_lr_epochs=cfg.update_lr_epochs,
            steps_per_epoch=dataset_size,
        )
    else:
        raise ValueError

    if cfg.arch == "bn_inception":
        grouped_parameters = get_param_groups(network.trainable_params())

        first_conv_weight = grouped_parameters[0]
        first_conv_bias = grouped_parameters[1]
        first_bn = grouped_parameters[2]

        normal_weight = grouped_parameters[3]
        normal_bias = grouped_parameters[4]

        optim_group_params = [
            {'params': first_conv_weight, 'lr': lr, 'weight_decay': cfg.weight_decay},
            {'params': first_conv_bias, 'lr': lr * 2, 'weight_decay': 0},
            {'params': first_bn, 'lr': lr, 'weight_decay': 0},

            {'params': normal_weight, 'lr': lr, 'weight_decay': cfg.weight_decay},
            {'params': normal_bias, 'lr': lr * 2, 'weight_decay': 0}
        ]
    elif cfg.arch == "van_base":
        optim_group_params = get_param_groups_van(network)
    else:
        raise NotImplementedError
    optimizer = nn.Momentum(learning_rate=lr, params=optim_group_params, momentum=cfg.momentum)
    return optimizer


def get_param_groups_van(network):
    """ get param groups """
    decay_params = []
    no_decay_params = []
    for x in network.trainable_params():
        parameter_name = x.name
        if parameter_name.endswith(".weight"):
            # Dense or Conv's weight using weight decay
            decay_params.append(x)
        else:
            # all bias not using weight decay
            # bn weight bias not using weight decay, be carefully for now x not include LN
            no_decay_params.append(x)

    return [{'params': no_decay_params, 'weight_decay': 0.0}, {'params': decay_params}]


def prepare_callbacks(cfg, network, dataset_size):
    """Prepare callbacks"""
    print("prepare callbacks...")

    callbacks = [
        TimeMonitor(data_size=dataset_size),
        LossMonitor(cfg.log_interval),
    ]

    checkpoint_config = CheckpointConfig(
        save_checkpoint_steps=cfg.ckpt_save_interval * dataset_size,
        keep_checkpoint_max=cfg.keep_checkpoint_max,
        saved_network=network,
    )

    checkpoint_output_dir_path = cfg.ckpt_file
    ckpt_save_callback = ModelCheckpoint(
        prefix="checkpoint_trn",
        config=checkpoint_config,
        directory=checkpoint_output_dir_path,
    )
    callbacks.append(ckpt_save_callback)
    if cfg.enable_modelarts:
        callbacks.append(MoxingCallBack(
            src_url=cfg.ckpt_file, train_url=os.path.join(cfg.train_url, f"ckpt_{cfg.rank}"),
            total_epochs=cfg.epochs_num, save_freq=cfg.keep_checkpoint_max))
    return callbacks


def run_train(cfg):
    """Run model train"""
    if "DEVICE_NUM" not in os.environ.keys():
        os.environ["DEVICE_NUM"] = str(cfg.device_num)
    if "RANK_SIZE" not in os.environ.keys():
        os.environ["RANK_SIZE"] = str(cfg.device_num)
    os.environ["DEVICE_TARGET"] = cfg.device_target
    prepare_context(cfg)

    logger = get_logger(cfg.train_output_dir, cfg.rank)
    logger.save_args(cfg)

    logger.important_info('Create the model')

    # Prepare the backbone
    if cfg.arch == "bn_inception":
        backbone = BNInception(out_channels=cfg.img_feature_dim, dropout=cfg.dropout, frozen_bn=True)
    elif cfg.arch == "van_base":
        backbone = van_base(num_classes=cfg.img_feature_dim)
    else:
        raise NotImplementedError
    initialize_backbone(cfg, backbone, cfg.pre_trained_backbone)

    if cfg.enable_modelarts:
        # you can edit copy code when it solved
        data_url = "/cache/dataset/trn_dataset.zip"
        sync_data("s3://open-data/pretrained/trn_dataset.zip", data_url)
        cfg.dataset_root = os.path.join(os.path.dirname(data_url), "trn_dataset")
    if cfg.arch == "bn_inception":
        mean = (104, 117, 128)
        std = (1, 1, 1)
    elif cfg.arch == "van_base":
        mean = [0.5 * 255, 0.5 * 255, 0.5 * 255]
        std = [0.5 * 255, 0.5 * 255, 0.5 * 255]
    else:
        raise NotImplementedError

    dataset, num_class = get_dataset_for_training(
        dataset_root=cfg.dataset_root,
        images_dir_name=cfg.images_dir_name,
        files_list_name=cfg.train_list_file_name,
        image_size=cfg.image_size,
        num_segments=cfg.num_segments,
        batch_size=cfg.train_batch_size,
        subsample_num=cfg.subsample_num,
        seed=cfg.seed,
        rank=cfg.rank,
        group_size=cfg.group_size,
        train_workers=cfg.train_workers,
        mean=mean, std=std
    )

    trn_head = RelationModuleMultiScale(
        cfg.img_feature_dim,
        cfg.num_segments,
        num_class,
        subsample_num=cfg.subsample_num,
    )
    network = TSN(base_network=backbone, consensus_network=trn_head)

    # cast amp_level for network
    cast_amp(network, amp_level=cfg.amp_level)

    net_optimizer = prepare_optimizer(cfg, network, dataset_size=dataset.get_dataset_size())

    # loss
    net_loss = CrossEntropySmooth(sparse=True, reduction='mean')
    if cfg.amp_level != "O0":
        scale_sense = nn.wrap.loss_scale.DynamicLossScaleUpdateCell(loss_scale_value=2 ** 10, scale_factor=2,
                                                                    scale_window=2000)
    else:
        scale_sense = Tensor(1., mstype.float32)

    net_with_loss = CustomWithLossCell(network, net_loss)
    net_loss_opt = CustomTrainOneStepCell(net_with_loss, net_optimizer, scale_sense=scale_sense,
                                          max_grad_norm=cfg.clip_grad_norm)

    cfg.ckpt_file = "./ckpt_" + str(cfg.rank)
    if cfg.enable_modelarts:
        cfg.ckpt_file = "/cache/ckpt_" + str(cfg.rank)
    print(f"=> cfg.ckpt_file: {cfg.ckpt_file}")
    # Callbacks
    callbacks = prepare_callbacks(cfg, network, dataset_size=dataset.get_dataset_size())

    # Creating a Model wrapper and training
    model = Model(net_loss_opt)

    logger.important_info('Train')
    logger.info('Total steps: %d', dataset.get_dataset_size())

    print("\n" + "*" * 70)
    model.train(cfg.epochs_num, dataset, callbacks=callbacks, dataset_sink_mode=True)

    if cfg.enable_modelarts:
        import moxing as mox
        mox.file.copy_parallel(src_url=cfg.ckpt_file,
                               dst_url=os.path.join(cfg.train_url, "ckpt_" + str(cfg.rank)))


class CrossEntropySmooth(LossBase):
    """CrossEntropy"""

    def __init__(self, sparse=True, reduction='mean'):
        super(CrossEntropySmooth, self).__init__()
        self.onehot = P.OneHot()
        self.sparse = sparse
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0., mstype.float32)
        self.ce = nn.SoftmaxCrossEntropyWithLogits(reduction=reduction)
        self.cast = ops.Cast()

    def construct(self, logit, label):
        if self.sparse:
            label = self.onehot(label, F.shape(logit)[1], self.on_value, self.off_value)
        label = P.Cast()(label, mstype.float32)
        logit = P.Cast()(logit, mstype.float32)
        loss2 = self.ce(logit, label)
        return loss2


if __name__ == '__main__':
    run_train(config)
