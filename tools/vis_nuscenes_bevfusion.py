import argparse
import glob
from pathlib import Path

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

from visual_utils import matplotlib_vis_utils as M

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate, build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--thresh', type=float, default=None, help='specify the score threshold for predicted boxes')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=1,
        logger=logger,
        training=False
    )
    logger.info(f'Total number of samples: \t{len(test_set)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(test_set):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = test_set.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            # print(data_dict.keys())
            boxes = pred_dicts[0]['pred_boxes']
            scores = pred_dicts[0]['pred_scores']
            if args.thresh:
                boxes = boxes[scores>=args.thresh]
            V.draw_scenes(
                points=data_dict['points'][:, 1:], ref_boxes=boxes, gt_boxes=data_dict['gt_boxes'][0,:,:9]
            )

            M.draw_bev_features(data_dict["img_feats_vis"][0], data_dict["lidar_feats_vis"][0],
                                     data_dict["fused_feats_vis"][0])

            if not OPEN3D_FLAG:
                mlab.show(stop=True)

            if idx==0:
                break

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
