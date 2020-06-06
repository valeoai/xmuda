import os
import os.path as osp
import shutil
import numpy as np
import pickle
import json
from PIL import Image
import cv2
import glob
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader


from xmuda.data.a2d2 import splits

from xmuda.data.a2d2.a2d2_dataloader import A2D2Base

# prevent "RuntimeError: received 0 items of ancdata"
torch.multiprocessing.set_sharing_strategy('file_system')


class_names_to_id = dict(zip(A2D2Base.class_names, range(len(A2D2Base.class_names))))


def undistort_image(config, image, cam_name):
    """copied from https://www.a2d2.audi/a2d2/en/tutorial.html"""
    if cam_name in ['front_left', 'front_center',
                    'front_right', 'side_left',
                    'side_right', 'rear_center']:
        # get parameters from config file
        intr_mat_undist = np.asarray(config['cameras'][cam_name]['CamMatrix'])
        intr_mat_dist = np.asarray(config['cameras'][cam_name]['CamMatrixOriginal'])
        dist_parms = np.asarray(config['cameras'][cam_name]['Distortion'])
        lens = config['cameras'][cam_name]['Lens']

        if lens == 'Fisheye':
            return cv2.fisheye.undistortImage(image, intr_mat_dist, D=dist_parms, Knew=intr_mat_undist)
        elif lens == 'Telecam':
            return cv2.undistort(image, intr_mat_dist, distCoeffs=dist_parms, newCameraMatrix=intr_mat_undist)
        else:
            return image
    else:
        return image


class DummyDataset(Dataset):
    """Use torch dataloader for multiprocessing"""
    def __init__(self, root_dir, scenes):
        self.class_names = A2D2Base.class_names.copy()
        self.categories = A2D2Base.categories.copy()
        self.root_dir = root_dir
        self.data = []
        self.glob_frames(scenes)

        # load config
        with open(osp.join(root_dir, 'cams_lidars.json'), 'r') as f:
            self.config = json.load(f)

        # load color to class mapping
        with open(osp.join(root_dir, 'class_list.json'), 'r') as f:
            class_list = json.load(f)
            self.rgb_to_class = {}
            self.rgb_to_cls_idx = {}
            count = 0
            for k, v in class_list.items():
                # hex to rgb
                rgb_value = tuple(int(k.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))
                self.rgb_to_class[rgb_value] = v
                self.rgb_to_cls_idx[rgb_value] = count
                count += 1

        assert list(class_names_to_id.keys()) == list(self.rgb_to_class.values())

    def glob_frames(self, scenes):
        for scene in scenes:
            cam_paths = sorted(glob.glob(osp.join(self.root_dir, scene, 'camera', 'cam_front_center', '*.png')))
            for cam_path in cam_paths:
                basename = osp.basename(cam_path)
                datetime = basename[:14]
                assert datetime.isdigit()
                frame_id = basename[-13:-4]
                assert frame_id.isdigit()
                data = {
                    'camera_path': cam_path,
                    'lidar_path': osp.join(self.root_dir, scene, 'lidar', 'cam_front_center',
                                           datetime + '_lidar_frontcenter_' + frame_id + '.npz'),
                    'label_path': osp.join(self.root_dir, scene, 'label', 'cam_front_center',
                                           datetime + '_label_frontcenter_' + frame_id + '.png'),
                }
                for k, v in data.items():
                    if not osp.exists(v):
                        raise IOError('File not found {}'.format(v))
                self.data.append(data)

    def __getitem__(self, index):
        data_dict = self.data[index].copy()
        lidar_front_center = np.load(data_dict['lidar_path'])
        points = lidar_front_center['points']
        if 'row' not in lidar_front_center.keys():
            print('row not in lidar dict, return None, {}'.format(data_dict['lidar_path']))
            return {}
        rows = lidar_front_center['row'].astype(np.int)
        cols = lidar_front_center['col'].astype(np.int)

        # extract 3D labels from 2D
        label_img = np.array(Image.open(data_dict['label_path']))
        label_img = undistort_image(self.config, label_img, 'front_center')
        label_pc = label_img[rows, cols, :]
        seg_label = np.full(label_pc.shape[0], fill_value=len(self.rgb_to_cls_idx), dtype=np.int64)
        # map RGB label code to index
        for rgb_values, cls_idx in self.rgb_to_cls_idx.items():
            idx = (rgb_values == label_pc).all(1)
            if idx.any():
                seg_label[idx] = cls_idx

        # load image
        image = Image.open(data_dict['camera_path'])
        image_size = image.size
        assert image_size == (1920, 1208)
        # undistort
        image = undistort_image(self.config, np.array(image), 'front_center')
        # scale image points
        points_img = np.stack([lidar_front_center['row'], lidar_front_center['col']], 1).astype(np.float32)
        # check if conversion from float64 to float32 has led to image points outside of image
        assert np.all(points_img[:, 0] < image_size[1])
        assert np.all(points_img[:, 1] < image_size[0])

        data_dict['seg_label'] = seg_label.astype(np.uint8)
        data_dict['points'] = points.astype(np.float32)
        data_dict['points_img'] = points_img  # row, col format, shape: (num_points, 2)
        data_dict['img'] = image

        return data_dict

    def __len__(self):
        return len(self.data)


def preprocess(split_name, root_dir, out_dir):
    pkl_data = []
    split = getattr(splits, split_name)

    dataloader = DataLoader(DummyDataset(root_dir, split), num_workers=8)

    num_skips = 0
    for i, data_dict in enumerate(dataloader):
        # data error leads to returning empty dict
        if not data_dict:
            print('empty dict, continue')
            num_skips += 1
            continue
        for k, v in data_dict.items():
            data_dict[k] = v[0]
        print('{}/{} {}'.format(i, len(dataloader), data_dict['lidar_path']))

        # convert to relative path
        lidar_path = data_dict['lidar_path'].replace(root_dir + '/', '')
        cam_path = data_dict['camera_path'].replace(root_dir + '/', '')

        # save undistorted image
        new_cam_path = osp.join(out_dir, cam_path)
        os.makedirs(osp.dirname(new_cam_path), exist_ok=True)
        image = Image.fromarray(data_dict['img'].numpy())
        image.save(new_cam_path)

        # append data
        out_dict = {
            'points': data_dict['points'].numpy(),
            'seg_labels': data_dict['seg_label'].numpy(),
            'points_img': data_dict['points_img'].numpy(),  # row, col format, shape: (num_points, 2)
            'lidar_path': lidar_path,
            'camera_path': cam_path,
        }
        pkl_data.append(out_dict)

    print('Skipped {} files'.format(num_skips))

    # save to pickle file
    save_dir = osp.join(out_dir, 'preprocess')
    os.makedirs(save_dir, exist_ok=True)
    save_path = osp.join(save_dir, '{}.pkl'.format(split_name))
    with open(save_path, 'wb') as f:
        pickle.dump(pkl_data, f)
        print('Wrote preprocessed data to ' + save_path)


if __name__ == '__main__':
    root_dir = '/datasets_master/a2d2'
    out_dir = '/datasets_local/datasets_mjaritz/a2d2_preprocess'
    preprocess('test', root_dir, out_dir)
    # split into train1 and train2 to prevent segmentation fault in torch dataloader
    preprocess('train1', root_dir, out_dir)
    preprocess('train2', root_dir, out_dir)
    # merge train1 and train2
    data = []
    for curr_split in ['train1', 'train2']:
        with open(osp.join(out_dir, 'preprocess', curr_split + '.pkl'), 'rb') as f:
            data.extend(pickle.load(f))
    save_path = osp.join(out_dir, 'preprocess', 'train.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
        print('Wrote preprocessed data to ' + save_path)
    for curr_split in ['train1', 'train2']:
        os.remove(osp.join(out_dir, 'preprocess', curr_split + '.pkl'))

    # copy cams_lidars.json and class_list.json to out_dir
    for filename in ['cams_lidars.json', 'class_list.json']:
        shutil.copyfile(osp.join(root_dir, filename), osp.join(out_dir, filename))