#!/usr/bin/env python
import os
import os.path as osp
import argparse
import logging
import time
import socket
import warnings

import torch

from xmuda.common.utils.checkpoint import CheckpointerV2
from xmuda.common.utils.logger import setup_logger
from xmuda.common.utils.metric_logger import MetricLogger
from xmuda.common.utils.torch_util import set_random_seed
from xmuda.models.build import build_model_2d, build_model_3d
from xmuda.data.build import build_dataloader
from xmuda.data.utils.validate import validate


def parse_args():
    parser = argparse.ArgumentParser(description='xMUDA test')
    parser.add_argument(
        '--cfg',
        dest='config_file',
        default='',
        metavar='FILE',
        help='path to config file',
        type=str,
    )
    parser.add_argument('ckpt2d', type=str, help='path to checkpoint file of the 2D model')
    parser.add_argument('ckpt3d', type=str, help='path to checkpoint file of the 3D model')
    parser.add_argument('--pselab', action='store_true', help='generate pseudo-labels')
    parser.add_argument(
        'opts',
        help='Modify config options using the command-line',
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    return args


def test(cfg, args, output_dir=''):
    logger = logging.getLogger('xmuda.test')

    # build 2d model
    model_2d = build_model_2d(cfg)[0]

    # build 3d model
    model_3d = build_model_3d(cfg)[0]

    model_2d = model_2d.cuda()
    model_3d = model_3d.cuda()

    # build checkpointer
    checkpointer_2d = CheckpointerV2(model_2d, save_dir=output_dir, logger=logger)
    if args.ckpt2d:
        # load weight if specified
        weight_path = args.ckpt2d.replace('@', output_dir)
        checkpointer_2d.load(weight_path, resume=False)
    else:
        # load last checkpoint
        checkpointer_2d.load(None, resume=True)
    checkpointer_3d = CheckpointerV2(model_3d, save_dir=output_dir, logger=logger)
    if args.ckpt3d:
        # load weight if specified
        weight_path = args.ckpt3d.replace('@', output_dir)
        checkpointer_3d.load(weight_path, resume=False)
    else:
        # load last checkpoint
        checkpointer_3d.load(None, resume=True)

    # build dataset
    test_dataloader = build_dataloader(cfg, mode='test', domain='target')

    pselab_path = None
    if args.pselab:
        pselab_dir = osp.join(output_dir, 'pselab_data')
        os.makedirs(pselab_dir, exist_ok=True)
        assert len(cfg.DATASET_TARGET.TEST) == 1
        pselab_path = osp.join(pselab_dir, cfg.DATASET_TARGET.TEST[0] + '.npy')

    # ---------------------------------------------------------------------------- #
    # Test
    # ---------------------------------------------------------------------------- #

    set_random_seed(cfg.RNG_SEED)
    test_metric_logger = MetricLogger(delimiter='  ')
    model_2d.eval()
    model_3d.eval()

    validate(cfg, model_2d, model_3d, test_dataloader, test_metric_logger, pselab_path=pselab_path)


def main():
    args = parse_args()

    # load the configuration
    # import on-the-fly to avoid overwriting cfg
    from xmuda.common.config import purge_cfg
    from xmuda.config.xmuda import cfg
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    purge_cfg(cfg)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    # replace '@' with config path
    if output_dir:
        config_path = osp.splitext(args.config_file)[0]
        output_dir = output_dir.replace('@', config_path.replace('configs/', ''))
        if not osp.isdir(output_dir):
            warnings.warn('Make a new directory: {}'.format(output_dir))
            os.makedirs(output_dir)

    # run name
    timestamp = time.strftime('%m-%d_%H-%M-%S')
    hostname = socket.gethostname()
    run_name = '{:s}.{:s}'.format(timestamp, hostname)

    logger = setup_logger('xmuda', output_dir, comment='test.{:s}'.format(run_name))
    logger.info('{:d} GPUs available'.format(torch.cuda.device_count()))
    logger.info(args)

    logger.info('Loaded configuration file {:s}'.format(args.config_file))
    logger.info('Running with config:\n{}'.format(cfg))

    assert cfg.MODEL_2D.DUAL_HEAD == cfg.MODEL_3D.DUAL_HEAD
    test(cfg, args, output_dir)


if __name__ == '__main__':
    main()
