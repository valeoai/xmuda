import numpy as np
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points

import matplotlib.pyplot as plt


# modified from https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/nuscenes.py
def map_pointcloud_to_image(pc, im_shape, info, im=None):
    """
    Maps the lidar point cloud to the image.
    :param pc: (3, N)
    :param im_shape: image to check size and debug
    :param info: dict with calibration infos
    :param im: image, only for visualization
    :return:
    """
    pc = pc.copy()

    # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
    # First step: transform the point-cloud to the ego vehicle frame for the timestamp of the sweep.
    pc = Quaternion(info['lidar2ego_rotation']).rotation_matrix @ pc
    pc = pc + np.array(info['lidar2ego_translation'])[:, np.newaxis]

    # Second step: transform to the global frame.
    pc = Quaternion(info['ego2global_rotation_lidar']).rotation_matrix @ pc
    pc = pc + np.array(info['ego2global_translation_lidar'])[:, np.newaxis]

    # Third step: transform into the ego vehicle frame for the timestamp of the image.
    pc = pc - np.array(info['ego2global_translation_cam'])[:, np.newaxis]
    pc = Quaternion(info['ego2global_rotation_cam']).rotation_matrix.T @ pc

    # Fourth step: transform into the camera.
    pc = pc - np.array(info['cam2ego_translation'])[:, np.newaxis]
    pc = Quaternion(info['cam2ego_rotation']).rotation_matrix.T @ pc

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depths = pc[2, :]

    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
    points = view_points(pc, np.array(info['cam_intrinsic']), normalize=True)

    # Cast to float32 to prevent later rounding errors
    points = points.astype(np.float32)

    # Remove points that are either outside or behind the camera.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > 0)
    mask = np.logical_and(mask, points[0, :] > 0)
    mask = np.logical_and(mask, points[0, :] < im_shape[1])
    mask = np.logical_and(mask, points[1, :] > 0)
    mask = np.logical_and(mask, points[1, :] < im_shape[0])
    points = points[:, mask]

    # debug
    if im is not None:
        # Retrieve the color from the depth.
        coloring = depths
        coloring = coloring[mask]

        plt.figure(figsize=(9, 16))
        plt.imshow(im)
        plt.scatter(points[0, :], points[1, :], c=coloring, s=2)
        plt.axis('off')

        # plt.show()

    return mask, pc.T, points.T[:, :2]
