"""
Open3d visualization tool box
Written by Jihan YANG
All rights preserved from 2021 - present.
"""
import open3d
import torch
import matplotlib
import numpy as np

box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
]


def get_coor_colors(obj_labels):
    """
    Args:
        obj_labels: 1 is ground, labels > 1 indicates different instance cluster

    Returns:
        rgb: [N, 3]. color for each point.
    """
    colors = matplotlib.colors.XKCD_COLORS.values()
    max_color_num = obj_labels.max()

    color_list = list(colors)[:max_color_num+1]
    colors_rgba = [matplotlib.colors.to_rgba_array(color) for color in color_list]
    label_rgba = np.array(colors_rgba)[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]

    return label_rgba


def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=True):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()

    vis = open3d.visualization.Visualizer()
    vis.create_window()

    if vis.get_render_option():
        vis.get_render_option().point_size = 1.0
        vis.get_render_option().background_color = np.zeros(3)

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(pts)
    if point_colors is None:
        pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)

    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (0, 1, 0))

    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, (1, 0, 0), ref_labels, ref_scores)

    vis.run()
    vis.destroy_window()


def draw_scenes_projected(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, img_dims=(720, 1080),
                          vfov=60.0, center=np.array([0, 0, 0]), eye=np.array([0, 0, 35]), up=np.array([0, 1, 0]),
                          near_clip=1.0, far_clip=80.0):

    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()

    renderer = open3d.visualization.rendering.OffscreenRenderer(*img_dims)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])
    colours = np.ones((points.shape[0], 3))
    colours[:,0] = 0
    pts.colors = open3d.utility.Vector3dVector(colours)

    material = open3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    material.point_size = 3.0

    renderer.scene.add_geometry("points", pts, material)

    if gt_boxes is not None:
        renderer = draw_box_proj(renderer, gt_boxes, (0, 1, 0), label="gt_bbox_lines")

    if ref_boxes is not None:
        renderer = draw_box_proj(renderer, ref_boxes, (1, 0, 0), ref_labels, ref_scores, label="ref_bbox_lines")

    renderer.setup_camera(vertical_field_of_view=vfov, center=center, eye=eye, up=up, near_clip=near_clip,
                          far_clip=far_clip)

    img = renderer.render_to_image()
    return img


def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d


def draw_box(vis, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None):
    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])
        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
            line_set.paint_uniform_color(box_colormap[ref_labels[i]])

        vis.add_geometry(line_set)
    return vis


def draw_box_proj(renderer, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None, label="bbox_lines"):
    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])
        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
            line_set.paint_uniform_color(box_colormap[ref_labels[i]])

        material = open3d.visualization.rendering.MaterialRecord()
        material.shader = "unlitLine"
        material.line_width = 2.0
        renderer.scene.add_geometry(label + f"_{i}", line_set, material)
    return renderer
