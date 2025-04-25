import tempfile

import pytest
import matplotlib.pyplot as plt
from tools.infer import main
import open3d


@pytest.fixture(autouse=True)
def disable_show(monkeypatch):
    monkeypatch.setattr(plt, 'show', lambda: None)
    monkeypatch.setattr(open3d.visualization.Visualizer, 'run', lambda x: None)
    monkeypatch.setattr(open3d.visualization.Visualizer, 'create_window', lambda x: None)


def test_valid_cam_jpg():
    args = ['--cfg_file=test_cfgs/bevfusion.yaml', '--ckpt=../pretrained_models/nuscenes_bevfusion.pth',
            '--data_path=test_data/test_data_cam_jpg', '--pc_fmt=pcd', '--im_fmt=jpg', '--thresh=0.05']
    main(args)


def test_valid_cam_np():
    args = ['--cfg_file=test_cfgs/bevfusion.yaml', '--ckpt=../pretrained_models/nuscenes_bevfusion.pth',
            '--data_path=test_data/test_data_cam_np', '--pc_fmt=pcd', '--im_fmt=np', '--thresh=0.05']
    main(args)


def test_valid_cam_png():
    args = ['--cfg_file=test_cfgs/bevfusion.yaml', '--ckpt=../pretrained_models/nuscenes_bevfusion.pth',
            '--data_path=test_data/test_data_cam_png', '--pc_fmt=pcd', '--im_fmt=png', '--thresh=0.05']
    main(args)


def test_valid_cam_pt():
    args = ['--cfg_file=test_cfgs/bevfusion.yaml', '--ckpt=../pretrained_models/nuscenes_bevfusion.pth',
            '--data_path=test_data/test_data_cam_pt', '--pc_fmt=pcd', '--im_fmt=pt', '--thresh=0.05']
    main(args)


def test_invalid_cam_small():
    args = ['--cfg_file=test_cfgs/bevfusion.yaml', '--ckpt=../pretrained_models/nuscenes_bevfusion.pth',
            '--data_path=test_data/test_data_cam_small', '--pc_fmt=pcd', '--im_fmt=jpg', '--thresh=0.05']
    with pytest.raises(Exception) as e_info:
        main(args)


def test_invalid_cam_views():
    args = ['--cfg_file=test_cfgs/bevfusion.yaml', '--ckpt=../pretrained_models/nuscenes_bevfusion.pth',
            '--data_path=test_data/test_data_cam_views', '--pc_fmt=pcd', '--im_fmt=jpg', '--thresh=0.05']
    with pytest.raises(Exception) as e_info:
        main(args)


def test_invalid_cam_format():
    args = ['--cfg_file=test_cfgs/bevfusion.yaml', '--ckpt=../pretrained_models/nuscenes_bevfusion.pth',
            '--data_path=test_data/test_data_cam_pdf', '--pc_fmt=pcd', '--im_fmt=pdf', '--thresh=0.05']
    with pytest.raises(SystemExit) as e_info:
        main(args)


def test_valid_lidar_np():
    args = ['--cfg_file=test_cfgs/bevfusion.yaml', '--ckpt=../pretrained_models/nuscenes_bevfusion.pth',
            '--data_path=test_data/test_data_lidar_np', '--pc_fmt=np', '--im_fmt=jpg', '--thresh=0.05']
    main(args)


def test_valid_lidar_pcd():
    args = ['--cfg_file=test_cfgs/bevfusion.yaml', '--ckpt=../pretrained_models/nuscenes_bevfusion.pth',
            '--data_path=test_data/test_data_lidar_pcd', '--pc_fmt=pcd', '--im_fmt=jpg', '--thresh=0.05']
    main(args)


def test_valid_lidar_pt():
    args = ['--cfg_file=test_cfgs/bevfusion.yaml', '--ckpt=../pretrained_models/nuscenes_bevfusion.pth',
            '--data_path=test_data/test_data_lidar_pt', '--pc_fmt=pt', '--im_fmt=jpg', '--thresh=0.05']
    main(args)


def test_invalid_lidar_format():
    args = ['--cfg_file=test_cfgs/bevfusion.yaml', '--ckpt=../pretrained_models/nuscenes_bevfusion.pth',
            '--data_path=test_data/test_data_lidar_csv', '--pc_fmt=csv', '--im_fmt=jpg', '--thresh=0.05']
    with pytest.raises(SystemExit) as e_info:
        main(args)
