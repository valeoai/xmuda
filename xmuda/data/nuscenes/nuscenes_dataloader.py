import os.path as osp
import pickle
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as T

from xmuda.data.utils.refine_pseudo_labels import refine_pseudo_labels
from xmuda.data.utils.augmentation_3d import augment_and_scale_3d


class NuScenesBase(Dataset):
    """NuScenes dataset"""

    class_names = [
        "car",
        "truck",
        "bus",
        "trailer",
        "construction_vehicle",
        "pedestrian",
        "motorcycle",
        "bicycle",
        "traffic_cone",
        "barrier",
        "background",
    ]

    # use those categories if merge_classes == True
    categories = {
        "vehicle": ["car", "truck", "bus", "trailer", "construction_vehicle"],
        "pedestrian": ["pedestrian"],
        "bike": ["motorcycle", "bicycle"],
        "traffic_boundary": ["traffic_cone", "barrier"],
        "background": ["background"]
    }

    def __init__(self,
                 split,
                 preprocess_dir,
                 merge_classes=False,
                 pselab_paths=None
                 ):

        self.split = split
        self.preprocess_dir = preprocess_dir

        print("Initialize Nuscenes dataloader")

        assert isinstance(split, tuple)
        print('Load', split)
        self.data = []
        for curr_split in split:
            with open(osp.join(self.preprocess_dir, curr_split + '.pkl'), 'rb') as f:
                self.data.extend(pickle.load(f))

        self.pselab_data = None
        if pselab_paths:
            assert isinstance(pselab_paths, tuple)
            print('Load pseudo label data ', pselab_paths)
            self.pselab_data = []
            for curr_split in pselab_paths:
                self.pselab_data.extend(np.load(curr_split, allow_pickle=True))

            # check consistency of data and pseudo labels
            assert len(self.pselab_data) == len(self.data)
            for i in range(len(self.pselab_data)):
                assert len(self.pselab_data[i]['pseudo_label_2d']) == len(self.data[i]['seg_labels'])

            # refine 2d pseudo labels
            probs2d = np.concatenate([data['probs_2d'] for data in self.pselab_data])
            pseudo_label_2d = np.concatenate([data['pseudo_label_2d'] for data in self.pselab_data]).astype(np.int)
            pseudo_label_2d = refine_pseudo_labels(probs2d, pseudo_label_2d)

            # refine 3d pseudo labels
            # fusion model has only one final prediction saved in probs_2d
            if 'probs_3d' in self.pselab_data[0].keys():
                probs3d = np.concatenate([data['probs_3d'] for data in self.pselab_data])
                pseudo_label_3d = np.concatenate([data['pseudo_label_3d'] for data in self.pselab_data]).astype(np.int)
                pseudo_label_3d = refine_pseudo_labels(probs3d, pseudo_label_3d)
            else:
                pseudo_label_3d = None

            # undo concat
            left_idx = 0
            for data_idx in range(len(self.pselab_data)):
                right_idx = left_idx + len(self.pselab_data[data_idx]['probs_2d'])
                self.pselab_data[data_idx]['pseudo_label_2d'] = pseudo_label_2d[left_idx:right_idx]
                if pseudo_label_3d is not None:
                    self.pselab_data[data_idx]['pseudo_label_3d'] = pseudo_label_3d[left_idx:right_idx]
                else:
                    self.pselab_data[data_idx]['pseudo_label_3d'] = None
                left_idx = right_idx

        if merge_classes:
            self.label_mapping = -100 * np.ones(len(self.class_names), dtype=int)
            for cat_idx, cat_list in enumerate(self.categories.values()):
                for class_name in cat_list:
                    self.label_mapping[self.class_names.index(class_name)] = cat_idx
            self.class_names = list(self.categories.keys())
        else:
            self.label_mapping = None

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)


class NuScenesSCN(NuScenesBase):
    def __init__(self,
                 split,
                 preprocess_dir,
                 nuscenes_dir='',
                 pselab_paths=None,
                 merge_classes=False,
                 scale=20,
                 full_scale=4096,
                 use_image=False,
                 resize=(400, 225),
                 image_normalizer=None,
                 noisy_rot=0.0,  # 3D augmentation
                 flip_x=0.0,  # 3D augmentation
                 rot_z=0.0,  # 3D augmentation
                 transl=False,  # 3D augmentation
                 fliplr=0.0,  # 2D augmentation
                 color_jitter=None,  # 2D augmentation
                 output_orig=False
                 ):
        super().__init__(split,
                         preprocess_dir,
                         merge_classes=merge_classes,
                         pselab_paths=pselab_paths)

        self.nuscenes_dir = nuscenes_dir
        self.output_orig = output_orig

        # point cloud parameters
        self.scale = scale
        self.full_scale = full_scale
        # 3D augmentation
        self.noisy_rot = noisy_rot
        self.flip_x = flip_x
        self.rot_z = rot_z
        self.transl = transl

        # image parameters
        self.use_image = use_image
        if self.use_image:
            self.resize = resize
            self.image_normalizer = image_normalizer

            # data augmentation
            self.fliplr = fliplr
            self.color_jitter = T.ColorJitter(*color_jitter) if color_jitter else None

    def __getitem__(self, index):
        data_dict = self.data[index]

        points = data_dict['points'].copy()
        seg_label = data_dict['seg_labels'].astype(np.int64)

        if self.label_mapping is not None:
            seg_label = self.label_mapping[seg_label]

        out_dict = {}

        keep_idx = np.ones(len(points), dtype=np.bool)
        if self.use_image:
            points_img = data_dict['points_img'].copy()
            img_path = osp.join(self.nuscenes_dir, data_dict['camera_path'])
            image = Image.open(img_path)

            if self.resize:
                if not image.size == self.resize:
                    # check if we do not enlarge downsized images
                    assert image.size[0] > self.resize[0]

                    # scale image points
                    points_img[:, 0] = float(self.resize[1]) / image.size[1] * np.floor(points_img[:, 0])
                    points_img[:, 1] = float(self.resize[0]) / image.size[0] * np.floor(points_img[:, 1])

                    # resize image
                    image = image.resize(self.resize, Image.BILINEAR)

            img_indices = points_img.astype(np.int64)

            assert np.all(img_indices[:, 0] >= 0)
            assert np.all(img_indices[:, 1] >= 0)
            assert np.all(img_indices[:, 0] < image.size[1])
            assert np.all(img_indices[:, 1] < image.size[0])

            # 2D augmentation
            if self.color_jitter is not None:
                image = self.color_jitter(image)
            # PIL to numpy
            image = np.array(image, dtype=np.float32, copy=False) / 255.
            # 2D augmentation
            if np.random.rand() < self.fliplr:
                image = np.ascontiguousarray(np.fliplr(image))
                img_indices[:, 1] = image.shape[1] - 1 - img_indices[:, 1]

            # normalize image
            if self.image_normalizer:
                mean, std = self.image_normalizer
                mean = np.asarray(mean, dtype=np.float32)
                std = np.asarray(std, dtype=np.float32)
                image = (image - mean) / std

            out_dict['img'] = np.moveaxis(image, -1, 0)
            out_dict['img_indices'] = img_indices

        # 3D data augmentation and scaling from points to voxel indices
        # nuscenes lidar coordinates: x (right), y (front), z (up)
        coords = augment_and_scale_3d(points, self.scale, self.full_scale, noisy_rot=self.noisy_rot,
                                      flip_x=self.flip_x, rot_z=self.rot_z, transl=self.transl)

        # cast to integer
        coords = coords.astype(np.int64)

        # only use voxels inside receptive field
        idxs = (coords.min(1) >= 0) * (coords.max(1) < self.full_scale)

        out_dict['coords'] = coords[idxs]
        out_dict['feats'] = np.ones([len(idxs), 1], np.float32)  # simply use 1 as feature
        out_dict['seg_label'] = seg_label[idxs]

        if self.use_image:
            out_dict['img_indices'] = out_dict['img_indices'][idxs]

        if self.pselab_data is not None:
            out_dict.update({
                'pseudo_label_2d': self.pselab_data[index]['pseudo_label_2d'][keep_idx][idxs],
                'pseudo_label_3d': self.pselab_data[index]['pseudo_label_3d'][keep_idx][idxs]
            })

        if self.output_orig:
            out_dict.update({
                'orig_seg_label': seg_label,
                'orig_points_idx': idxs,
            })

        return out_dict


def test_NuScenesSCN():
    from xmuda.data.utils.visualize import draw_points_image_labels, draw_points_image_depth, draw_bird_eye_view
    preprocess_dir = '/datasets_local/datasets_mjaritz/nuscenes_preprocess/preprocess'
    nuscenes_dir = '/datasets_local/datasets_mjaritz/nuscenes_preprocess'
    # split = ('train_singapore',)
    # pselab_paths = ('/home/docker_user/workspace/outputs/xmuda/nuscenes/usa_singapore/xmuda/pselab_data/train_singapore.npy',)
    split = ('train_night',)
    # pselab_paths = ('/home/docker_user/workspace/outputs/xmuda/nuscenes/day_night/xmuda/pselab_data/train_night.npy',)
    dataset = NuScenesSCN(split=split,
                          preprocess_dir=preprocess_dir,
                          nuscenes_dir=nuscenes_dir,
                          # pselab_paths=pselab_paths,
                          merge_classes=True,
                          use_image=True,
                          noisy_rot=0.1,
                          flip_x=0.5,
                          rot_z=2*np.pi,
                          transl=True,
                          fliplr=0.5,
                          color_jitter=(0.4, 0.4, 0.4)
                          )
    for i in [10, 20, 30, 40, 50, 60]:
        data = dataset[i]
        coords = data['coords']
        seg_label = data['seg_label']
        img = np.moveaxis(data['img'], 0, 2)
        img_indices = data['img_indices']
        draw_points_image_labels(img, img_indices, seg_label, color_palette_type='NuScenes', point_size=3)
        # pseudo_label_2d = data['pseudo_label_2d']
        # draw_points_image_labels(img, img_indices, pseudo_label_2d, color_palette_type='NuScenes', point_size=3)
        draw_bird_eye_view(coords)
        print('Number of points:', len(coords))


def compute_class_weights():
    preprocess_dir = '/datasets_local/datasets_mjaritz/nuscenes_preprocess/preprocess'
    # split = ('train_usa', 'test_usa')
    split = ('train_day', 'test_day')
    dataset = NuScenesBase(split,
                           preprocess_dir,
                           merge_classes=True
                           )
    # compute points per class over whole dataset
    num_classes = len(dataset.class_names)
    points_per_class = np.zeros(num_classes, int)
    for i, data in enumerate(dataset.data):
        print('{}/{}'.format(i, len(dataset)))
        points_per_class += np.bincount(dataset.label_mapping[data['seg_labels']], minlength=num_classes)

    # compute log smoothed class weights
    class_weights = np.log(5 * points_per_class.sum() / points_per_class)
    print('log smoothed class weights: ', class_weights / class_weights.min())


if __name__ == '__main__':
    test_NuScenesSCN()
    # compute_class_weights()
