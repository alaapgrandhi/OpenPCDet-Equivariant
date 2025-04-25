import argparse
from pathlib import Path

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils
from PIL import Image
import os
import pandas as pd


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--out_path', type=str, default='sample_data', help='specify where to create the sample data')
    parser.add_argument('--index', type=int, default=0,
                        help='specify which sample from the NuScenes data you want to use to generate the sample data')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def save_cam_params(lid2cam, lid2im, cam2ego, cam_intr, cam2lid, out_path):
    np.save(out_path / "lid2cam.npy", lid2cam)
    np.save(out_path / "lid2im.npy", lid2im)
    np.save(out_path / "cam2ego.npy", cam2ego)
    np.save(out_path / "cam_intr.npy", cam_intr)
    np.save(out_path / "cam2lid.npy", cam2lid)


def save_imgs(cam_imgs, filetype, out_path):
    if filetype=="png":
        for idx, img in enumerate(cam_imgs):
            img.save(out_path / "images" / f"image_{idx}.png")
    elif filetype=="jpg":
        for idx, img in enumerate(cam_imgs):
            img.save(out_path / "images" / f"image_{idx}.jpg")
    elif filetype=="np":
        np_imgs = np.stack([np.asarray(img) for img in cam_imgs])
        np.save(out_path / "images" / "images.npy", np_imgs)
    elif filetype=="pt":
        torch_imgs = torch.tensor(np.stack([np.asarray(img) for img in cam_imgs]))
        torch.save(torch_imgs, out_path / "images" / "images.pt")
    elif filetype=="pdf":
        cam_imgs[0].save(out_path / "images" / "images.pdf", save_all=True, append_images=cam_imgs[1:])


def save_lidar(lid_pts, filetype, out_path):
    if filetype=="pcd":
        lid_pts.tofile(out_path / "pointcloud.pcd.bin")
    elif filetype=="np":
        np.save(out_path / "pointcloud.npy", lid_pts)
    elif filetype=="pt":
        torch.save(torch.tensor(lid_pts), out_path / "pointcloud.pt")
    elif filetype=="csv":
        df = pd.DataFrame(lid_pts)
        df.to_csv(out_path / "pointcloud.csv")


def create_sample_data(cam_imgs, lid_pts, lid2cam, lid2im, cam2ego, cam_intr, cam2lid, out_path):
    for im_type in ["png", "jpg", "np", "pt"]:
        temp_path = out_path / f"sample_data_cam_{im_type}"
        os.makedirs(temp_path / "images", exist_ok=True)
        save_cam_params(lid2cam, lid2im, cam2ego, cam_intr, cam2lid, temp_path)
        save_imgs(cam_imgs, im_type, temp_path)
        save_lidar(lid_pts, "pcd", temp_path)

    for lid_type in ["pcd", "np", "pt"]:
        temp_path = out_path / f"sample_data_lidar_{lid_type}"
        os.makedirs(temp_path / "images", exist_ok=True)
        save_cam_params(lid2cam, lid2im, cam2ego, cam_intr, cam2lid, temp_path)
        save_imgs(cam_imgs, "jpg", temp_path)
        save_lidar(lid_pts, lid_type, temp_path)

    temp_path = out_path / "sample_data_cam_pdf"
    os.makedirs(temp_path / "images", exist_ok=True)
    save_cam_params(lid2cam, lid2im, cam2ego, cam_intr, cam2lid, temp_path)
    save_imgs(cam_imgs, "pdf", temp_path)
    save_lidar(lid_pts, "pcd", temp_path)

    temp_path = out_path / "sample_data_cam_small"
    os.makedirs(temp_path / "images", exist_ok=True)
    save_cam_params(lid2cam, lid2im, cam2ego, cam_intr, cam2lid, temp_path)
    bad_cam_imgs = [img.resize((1, 1)) for img in cam_imgs]
    save_imgs(bad_cam_imgs, "jpg", temp_path)
    save_lidar(lid_pts, "pcd", temp_path)

    temp_path = out_path / "sample_data_cam_wrongnumviews"
    os.makedirs(temp_path / "images", exist_ok=True)
    save_cam_params(lid2cam, lid2im, cam2ego, cam_intr, cam2lid, temp_path)
    bad_cam_imgs = cam_imgs + cam_imgs[:4]
    save_imgs(bad_cam_imgs, "jpg", temp_path)
    save_lidar(lid_pts, "pcd", temp_path)

    temp_path = out_path / "sample_data_lidar_csv"
    os.makedirs(temp_path / "images", exist_ok=True)
    save_cam_params(lid2cam, lid2im, cam2ego, cam_intr, cam2lid, temp_path)
    save_imgs(cam_imgs, "jpg", temp_path)
    save_lidar(lid_pts, "csv", temp_path)

def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
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

    a = test_set[args.index]
    root_path = Path(cfg.DATA_CONFIG.DATA_PATH) / cfg.DATA_CONFIG.VERSION
    lid2cam = np.stack(a['lidar2camera'])
    lid2im = np.stack(a['lidar2image'])
    cam2ego = np.stack(a['camera2ego'])
    cam_intr = np.stack(a['camera_intrinsics'])
    cam2lid = np.stack(a['camera2lidar'])
    lid_pts = a['points']
    im_paths = [root_path / im_path for im_path in a['image_paths']]
    cam_ims = [Image.open(str(im_path)) for im_path in im_paths]
    create_sample_data(cam_ims, lid_pts, lid2cam, lid2im, cam2ego, cam_intr, cam2lid, Path(args.out_path))



if __name__ == '__main__':
    main()
