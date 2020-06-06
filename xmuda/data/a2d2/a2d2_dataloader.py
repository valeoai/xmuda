import os.path as osp
import pickle
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as T
import json

from xmuda.data.utils.augmentation_3d import augment_and_scale_3d


class A2D2Base(Dataset):
    """A2D2 dataset"""

    class_names = [
        'Car 1',
        'Car 2',
        'Car 3',
        'Car 4',
        'Bicycle 1',
        'Bicycle 2',
        'Bicycle 3',
        'Bicycle 4',
        'Pedestrian 1',
        'Pedestrian 2',
        'Pedestrian 3',
        'Truck 1',
        'Truck 2',
        'Truck 3',
        'Small vehicles 1',
        'Small vehicles 2',
        'Small vehicles 3',
        'Traffic signal 1',
        'Traffic signal 2',
        'Traffic signal 3',
        'Traffic sign 1',
        'Traffic sign 2',
        'Traffic sign 3',
        'Utility vehicle 1',
        'Utility vehicle 2',
        'Sidebars',
        'Speed bumper',
        'Curbstone',
        'Solid line',
        'Irrelevant signs',
        'Road blocks',
        'Tractor',
        'Non-drivable street',
        'Zebra crossing',
        'Obstacles / trash',
        'Poles',
        'RD restricted area',
        'Animals',
        'Grid structure',
        'Signal corpus',
        'Drivable cobblestone',
        'Electronic traffic',
        'Slow drive area',
        'Nature object',
        'Parking area',
        'Sidewalk',
        'Ego car',
        'Painted driv. instr.',
        'Traffic guide obj.',
        'Dashed line',
        'RD normal street',
        'Sky',
        'Buildings',
        'Blurred area',
        'Rain dirt'
    ]

    # use those categories if merge_classes == True
    categories = {
        'car': ['Car 1', 'Car 2', 'Car 3', 'Car 4', 'Ego car'],
        'truck': ['Truck 1', 'Truck 2', 'Truck 3'],
        'bike': ['Bicycle 1', 'Bicycle 2', 'Bicycle 3', 'Bicycle 4', 'Small vehicles 1', 'Small vehicles 2',
                 'Small vehicles 3'],  # small vehicles are "usually" motorcycles
        'person': ['Pedestrian 1', 'Pedestrian 2', 'Pedestrian 3'],
        'road': ['RD normal street', 'Zebra crossing', 'Solid line', 'RD restricted area', 'Slow drive area',
                 'Drivable cobblestone', 'Dashed line', 'Painted driv. instr.'],
        'parking': ['Parking area'],
        'sidewalk': ['Sidewalk', 'Curbstone'],
        'building': ['Buildings'],
        'nature': ['Nature object'],
        'other-objects': ['Poles', 'Traffic signal 1', 'Traffic signal 2', 'Traffic signal 3', 'Traffic sign 1',
                          'Traffic sign 2', 'Traffic sign 3', 'Sidebars', 'Speed bumper', 'Irrelevant signs',
                          'Road blocks', 'Obstacles / trash', 'Animals', 'Signal corpus', 'Electronic traffic',
                          'Traffic guide obj.', 'Grid structure'],
        # 'ignore': ['Sky', 'Utility vehicle 1', 'Utility vehicle 2', 'Tractor', 'Non-drivable street',
        #            'Blurred area', 'Rain dirt'],
    }

    def __init__(self,
                 split,
                 preprocess_dir,
                 merge_classes=False
                 ):

        self.split = split
        self.preprocess_dir = preprocess_dir

        print("Initialize A2D2 dataloader")

        with open(osp.join(self.preprocess_dir, 'cams_lidars.json'), 'r') as f:
            self.config = json.load(f)

        assert isinstance(split, tuple)
        print('Load', split)
        self.data = []
        for curr_split in split:
            with open(osp.join(self.preprocess_dir, 'preprocess', curr_split + '.pkl'), 'rb') as f:
                self.data.extend(pickle.load(f))

        with open(osp.join(self.preprocess_dir, 'class_list.json'), 'r') as f:
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

        assert self.class_names == list(self.rgb_to_class.values())
        if merge_classes:
            self.label_mapping = -100 * np.ones(len(self.rgb_to_class) + 1, dtype=int)
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


class A2D2SCN(A2D2Base):
    def __init__(self,
                 split,
                 preprocess_dir,
                 merge_classes=False,
                 scale=20,
                 full_scale=4096,
                 use_image=False,
                 resize=(480, 302),
                 image_normalizer=None,
                 noisy_rot=0.0,  # 3D augmentation
                 flip_y=0.0,  # 3D augmentation
                 rot_z=0.0,  # 3D augmentation
                 transl=False,  # 3D augmentation
                 fliplr=0.0,  # 2D augmentation
                 color_jitter=None,  # 2D augmentation
                 ):
        super().__init__(split,
                         preprocess_dir,
                         merge_classes=merge_classes)

        # point cloud parameters
        self.scale = scale
        self.full_scale = full_scale
        # 3D augmentation
        self.noisy_rot = noisy_rot
        self.flip_y = flip_y
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

        if self.use_image:
            points_img = data_dict['points_img'].copy()
            img_path = osp.join(self.preprocess_dir, data_dict['camera_path'])
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
        # A2D2 lidar coordinates (same as Kitti): x (front), y (left), z (up)
        coords = augment_and_scale_3d(points, self.scale, self.full_scale, noisy_rot=self.noisy_rot,
                                      flip_y=self.flip_y, rot_z=self.rot_z, transl=self.transl)

        # cast to integer
        coords = coords.astype(np.int64)

        # only use voxels inside receptive field
        idxs = (coords.min(1) >= 0) * (coords.max(1) < self.full_scale)

        out_dict['coords'] = coords[idxs]
        out_dict['feats'] = np.ones([len(idxs), 1], np.float32)  # simply use 1 as feature
        out_dict['seg_label'] = seg_label[idxs]

        if self.use_image:
            out_dict['img_indices'] = out_dict['img_indices'][idxs]

        return out_dict


def test_A2D2SCN():
    from xmuda.data.utils.visualize import draw_points_image_labels, draw_bird_eye_view
    preprocess_dir = '/datasets_local/datasets_mjaritz/a2d2_preprocess'
    split = ('test',)
    dataset = A2D2SCN(split=split,
                      preprocess_dir=preprocess_dir,
                      merge_classes=True,
                      use_image=True,
                      noisy_rot=0.1,
                      flip_y=0.5,
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
        draw_points_image_labels(img, img_indices, seg_label, color_palette_type='SemanticKITTI', point_size=3)
        draw_bird_eye_view(coords)


def compute_class_weights():
    preprocess_dir = '/datasets_local/datasets_mjaritz/a2d2_preprocess'
    split = ('train', 'test')
    dataset = A2D2Base(split,
                       preprocess_dir,
                       merge_classes=True
                       )
    # compute points per class over whole dataset
    num_classes = len(dataset.class_names)
    points_per_class = np.zeros(num_classes, int)
    for i, data in enumerate(dataset.data):
        print('{}/{}'.format(i, len(dataset)))
        labels = dataset.label_mapping[data['seg_labels']]
        points_per_class += np.bincount(labels[labels != -100], minlength=num_classes)

    # compute log smoothed class weights
    class_weights = np.log(5 * points_per_class.sum() / points_per_class)
    print('log smoothed class weights: ', class_weights / class_weights.min())


if __name__ == '__main__':
    test_A2D2SCN()
    # compute_class_weights()
