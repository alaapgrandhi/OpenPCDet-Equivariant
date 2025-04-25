import argparse
from easydict import EasyDict

from tools.visual_utils import open3d_vis_utils as V
from tools.visual_utils import matplotlib_vis_utils as M

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.datasets.processor.data_processor import DataProcessor
from pcdet.datasets.processor.point_feature_encoder import PointFeatureEncoder
from PIL import Image
import os


def parse_config(opt_args=None):
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data', help='specify the data directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--thresh', type=float, default=None, help='specify the score threshold for predicted boxes')
    parser.add_argument('--pc_fmt', type=str, default='pcd', help='specify the format of your point cloud data file',
                        choices=['pcd', 'np', 'pt'])
    parser.add_argument('--im_fmt', type=str, default='jpg', help='specify the format of your camera image file(s)',
                        choices=['png', 'jpg', 'np', 'pt'])
    args = parser.parse_args(opt_args)

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def crop_imgs(input_dict, camera_image_config):
    W, H = input_dict["ori_shape"]
    imgs = input_dict["camera_imgs"]
    img_process_infos = []
    crop_images = []
    for img in imgs:
        fH, fW = camera_image_config.FINAL_DIM
        resize_lim = camera_image_config.RESIZE_LIM_TEST
        resize = np.mean(resize_lim)
        resize_dims = (int(W * resize), int(H * resize))
        newW, newH = resize_dims
        crop_h = newH - fH
        crop_w = int(max(0, newW - fW) / 2)
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)

        # reisze and crop image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        crop_images.append(img)
        img_process_infos.append([resize, crop, False, 0])

    input_dict['img_process_infos'] = img_process_infos
    input_dict['camera_imgs'] = crop_images
    return input_dict


def load_data(args, cfg):
    lid_pts = None
    if args.pc_fmt == "pcd":
        lid_pts = np.fromfile(args.data_path + "/pointcloud.pcd.bin", dtype=np.float32).reshape(-1, 5)
    elif args.pc_fmt == "np":
        lid_pts = np.load(args.data_path + "/pointcloud.npy")
    elif args.pc_fmt == "pt":
        lid_pts = torch.load(args.data_path + "/pointcloud.pt").numpy()

    images = []
    if args.im_fmt == "png":
        assert not os.path.exists(args.data_path + f"/images/image_{6}.png")
        for i in range(6):
            images.append(Image.open(args.data_path + f"/images/image_{i}.png"))
    elif args.im_fmt == "jpg":
        assert not os.path.exists(args.data_path + f"/images/image_{6}.jpg")
        for i in range(6):
            images.append(Image.open(args.data_path + f"/images/image_{i}.jpg"))
    elif args.im_fmt == "np":
        np_images = np.load(args.data_path + "/images/images.npy")
        assert np_images.shape[0] == 6
        for i in range(6):
            images.append(Image.fromarray(np_images[i]))
    elif args.im_fmt == "pt":
        np_images = torch.load(args.data_path + "/images/images.pt").numpy()
        assert np_images.shape[0] == 6
        for i in range(6):
            images.append(Image.fromarray(np_images[i]))

    lid2cam = np.load(args.data_path + "/lid2cam.npy")
    lid2im = np.load(args.data_path + "/lid2im.npy")
    cam2ego = np.load(args.data_path + "/cam2ego.npy")
    cam_intr = np.load(args.data_path + "/cam_intr.npy")
    cam2lid = np.load(args.data_path + "/cam2lid.npy")

    for i in range(len(images)):
        width, height = images[i].size
        assert 24 <= width <= 4096
        assert 24 <= height <= 4096

    input_dict = {
        "points": lid_pts,
        "camera_imgs": images,
        "ori_shape": images[0].size,
        "lidar2camera": lid2cam,
        "lidar2image": lid2im,
        "camera2ego": cam2ego,
        "camera_intrinsics": cam_intr,
        "camera2lidar": cam2lid,
        "lidar_aug_matrix": np.eye(4)
    }

    input_dict = crop_imgs(input_dict, cfg.DATA_CONFIG.CAMERA_CONFIG.IMAGE)
    return input_dict


def load_processor_and_model(args, cfg):
    logger = common_utils.create_logger()

    pc_range = np.array(cfg.DATA_CONFIG.POINT_CLOUD_RANGE, dtype=np.float32)

    point_feature_encoder = PointFeatureEncoder(cfg.DATA_CONFIG.POINT_FEATURE_ENCODING, point_cloud_range=pc_range)

    data_processor = DataProcessor(
        cfg.DATA_CONFIG.DATA_PROCESSOR, point_cloud_range=pc_range,
        training=False, num_point_features=point_feature_encoder.num_point_features
    )

    if hasattr(data_processor, "depth_downsample_factor"):
        depth_downsample_factor = data_processor.depth_downsample_factor
    else:
        depth_downsample_factor = None

    dataset_placeholder = EasyDict({"class_names": cfg.CLASS_NAMES, "point_feature_encoder": point_feature_encoder,
                                    "grid_size": data_processor.grid_size, "point_cloud_range": pc_range,
                                    "voxel_size": data_processor.voxel_size,
                                    "depth_downsample_factor": depth_downsample_factor})

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset_placeholder)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    return point_feature_encoder, data_processor, model


def main(opt_args=None, return_vis=False, img_dims=(720, 1080), vfov=60.0, center=np.array([0, 0, 0]),
         eye=np.array([0, 0, 35]), up=np.array([0, 1, 0]), near_clip=1.0, far_clip=80.0):
    args, cfg = parse_config(opt_args)
    point_feature_encoder, data_processor, model = load_processor_and_model(args, cfg)
    input_dict = load_data(args, cfg)
    input_dict = point_feature_encoder.forward(input_dict)
    input_dict = data_processor.forward(input_dict)
    data_dict = DatasetTemplate.collate_batch([input_dict])
    load_data_to_gpu(data_dict)

    with torch.no_grad():
        pred_dicts, _ = model.forward(data_dict)
        boxes = pred_dicts[0]['pred_boxes']
        scores = pred_dicts[0]['pred_scores']
        if args.thresh:
            boxes = boxes[scores >= args.thresh]

        if not return_vis:
            V.draw_scenes(
                points=data_dict['points'][:, 1:], ref_boxes=boxes
            )

            M.draw_bev_features(data_dict["img_feats_vis"][0], data_dict["lidar_feats_vis"][0],
                                data_dict["fused_feats_vis"][0])
        else:
            img = V.draw_scenes_projected(points=data_dict['points'][:, 1:],
                                    ref_boxes=boxes,
                                    img_dims=img_dims,
                                    vfov=vfov,
                                    center=center,
                                    eye=eye,
                                    up=up,
                                    near_clip=near_clip,
                                    far_clip=far_clip
                                    )
            return img


if __name__ == '__main__':
    main()
