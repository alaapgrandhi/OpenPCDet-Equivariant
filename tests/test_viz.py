from matplotlib.testing.compare import compare_images
import tools.visual_utils.open3d_vis_utils as V
from tools.infer import main
import pytest
import tempfile
import open3d
import numpy as np
import matplotlib.pyplot as plt


def test_bev_view():
    pts = np.load("test_data/test_data_lidar_np/pointcloud.npy")
    bboxes = np.array([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0], [5.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0],
                       [0.0, 15.0, 0.0, 2.0, 2.0, 2.0, 0.0], [2.0, -5.0, 0.0, 1.0, 1.0, 1.0, 0.0]])
    img = V.draw_scenes_projected(points=pts, ref_boxes=bboxes)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmpfile:
        open3d.io.write_image(tmpfile.name, img)
        diff = compare_images("test_data/BEV_View.png", tmpfile.name, 1.0, in_decorator=True)

    if diff:
        pytest.fail("The rendered image did not match the version stored in test_data/BEV_View.png: " +
                    f"RMS={diff['rms']:.4f}")


def test_infer():
    args = ['--cfg_file=test_cfgs/dummy.yaml', '--ckpt=../pretrained_models/nuscenes_bevfusion.pth',
            '--data_path=test_data/test_data_cam_jpg', '--pc_fmt=pcd', '--im_fmt=jpg', '--thresh=0.05']
    img = main(args, return_vis=True)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmpfile:
        open3d.io.write_image(tmpfile.name, img)
        diff = compare_images("test_data/BEV_View_mean.png", tmpfile.name, 1.0, in_decorator=True)

    if diff:
        pytest.fail("The rendered image did not match the version stored in test_data/BEV_View.png: " +
                    f"RMS={diff['rms']:.4f}")
