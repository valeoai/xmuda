## [Updated code](https://github.com/valeoai/xmuda_journal) from our TPAMI paper.

# xMUDA: Cross-Modal Unsupervised Domain Adaptation for 3D Semantic Segmentation

Official code for the paper.

## Paper
![](./teaser.png)

[xMUDA: Cross-Modal Unsupervised Domain Adaptation for 3D Semantic Segmentation](https://arxiv.org/abs/1911.12676)  
 [Maximilian Jaritz](https://team.inria.fr/rits/membres/maximilian-jaritz/), [Tuan-Hung Vu](https://tuanhungvu.github.io/), [Raoul de Charette](https://team.inria.fr/rits/membres/raoul-de-charette/),  Émilie Wirbel, [Patrick Pérez](https://ptrckprz.github.io/)  
 Inria, valeo.ai
 CVPR 2020

If you find this code useful for your research, please cite our [paper](https://arxiv.org/abs/1911.12676):

```
@inproceedings{jaritz2019xmuda,
	title={{xMUDA}: Cross-Modal Unsupervised Domain Adaptation for {3D} Semantic Segmentation},
	author={Jaritz, Maximilian and Vu, Tuan-Hung and de Charette, Raoul and Wirbel, Emilie and P{\'e}rez, Patrick},
	booktitle={CVPR},
	year={2020}
}
```
## Preparation
### Prerequisites
Tested with
* PyTorch 1.4
* CUDA 10.0
* Python 3.8
* [SparseConvNet](https://github.com/facebookresearch/SparseConvNet)
* [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit)

### Installation
As 3D network we use SparseConvNet. It requires to use CUDA 10.0 (it did not work with 10.1 when we tried).
We advise to create a new conda environment for installation. PyTorch and CUDA can be installed, and SparseConvNet
installed/compiled as follows:
```
$ conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
$ pip install --upgrade git+https://github.com/facebookresearch/SparseConvNet.git
```

Clone this repository and install it with pip. It will automatically install the nuscenes-devkit as a dependency.
```
$ git clone https://github.com/valeoai/xmuda.git
$ cd xmuda
$ pip install -ve .
```
The `-e` option means that you can edit the code on the fly.

### Datasets
#### NuScenes
Please download the Full dataset (v1.0) from the [NuScenes website](https://www.nuscenes.org) and extract it.

You need to perform preprocessing to generate the data for xMUDA first.
The preprocessing subsamples the 360° LiDAR point cloud to only keep the points that project into
the front camera image. It also generates the point-wise segmentation labels using
the 3D objects by checking which points lie inside the 3D boxes. 
All information will be stored in a pickle file (except the images which will be 
read frame by frame by the dataloader during training).

Please edit the script `xmuda/data/nuscenes/preprocess.py` as follows and then run it.
* `root_dir` should point to the root directory of the NuScenes dataset
* `out_dir` should point to the desired output directory to store the pickle files

#### A2D2
Please download the Semantic Segmentation dataset and Sensor Configuration from the
[Audi website](https://www.a2d2.audi/a2d2/en/download.html) or directly use `wget` and
the following links, then extract.
```
$ wget https://aev-autonomous-driving-dataset.s3.eu-central-1.amazonaws.com/camera_lidar_semantic.tar
$ wget https://aev-autonomous-driving-dataset.s3.eu-central-1.amazonaws.com/cams_lidars.json
```

The dataset directory should have this basic structure:
```
a2d2                                   % A2D2 dataset root
 ├── 20180807_145028
 ├── 20180810_142822
 ├── ...
 ├── cams_lidars.json
 └── class_list.json
```
For preprocessing, we undistort the images and store them separately as .png files.
Similar to NuScenes preprocessing, we save all points that project into the front camera image as well
as the segmentation labels to a pickle file.

Please edit the script `xmuda/data/a2d2/preprocess.py` as follows and then run it.
* `root_dir` should point to the root directory of the A2D2 dataset
* `out_dir` should point to the desired output directory to store the undistorted images and pickle files.
It should be set differently than the `root_dir` to prevent overwriting of images.

#### SemanticKITTI
Please download the files from the [SemanticKITTI website](http://semantic-kitti.org/dataset.html) and
additionally the [color data](http://www.cvlibs.net/download.php?file=data_odometry_color.zip)
from the [Kitti Odometry website](http://www.cvlibs.net/datasets/kitti/eval_odometry.php). Extract
everything into the same folder.

Similar to NuScenes preprocessing, we save all points that project into the front camera image as well
as the segmentation labels to a pickle file.

Please edit the script `xmuda/data/semantic_kitti/preprocess.py` as follows and then run it.
* `root_dir` should point to the root directory of the SemanticKITTI dataset
* `out_dir` should point to the desired output directory to store the pickle files

## Training
### xMUDA
You can run the training with
```
$ cd <root dir of this repo>
$ python xmuda/train_xmuda.py --cfg=configs/nuscenes/usa_singapore/xmuda.yaml
```

The output will be written to `/home/<user>/workspace/outputs/xmuda/<config_path>` by 
default. The `OUTPUT_DIR` can be modified in the config file in
(e.g. `configs/nuscenes/usa_singapore/xmuda.yaml`) or optionally at run time in the
command line (dominates over config file). Note that `@` in the following example will be
automatically replaced with the config path, i.e. with `nuscenes/usa_singapore/xmuda`.
```
$ python xmuda/train_xmuda.py --cfg=configs/nuscenes/usa_singapore/xmuda.yaml OUTPUT_DIR path/to/output/directory/@
```

You can start the trainings on the other UDA scenarios (Day/Night and A2D2/SemanticKITTI) analogously:
```
$ python xmuda/train_xmuda.py --cfg=configs/nuscenes/day_night/xmuda.yaml
$ python xmuda/train_xmuda.py --cfg=configs/a2d2_semantic_kitti/xmuda.yaml
```

### xMUDA<sub>PL</sub>
After having trained the xMUDA model, generate the pseudo-labels as follows:
```
$ python xmuda/test.py --cfg=configs/nuscenes/usa_singapore/xmuda.yaml --pselab @/model_2d_100000.pth @/model_3d_100000.pth DATASET_TARGET.TEST "('train_singapore',)"
```
Note that we use the last model at 100,000 steps to exclude supervision from the validation set by picking the best
weights. The pseudo labels and maximum probabilities are saved as `.npy` file.

Please edit the `pselab_paths` in the config file, e.g. `configs/nuscenes/usa_singapore/xmuda_pl.yaml`,
to match your path of the generated pseudo-labels.

Then start the training. The pseudo-label refinement (discard less confident pseudo-labels) is done
when the dataloader is initialized.
```
$ python xmuda/train_xmuda.py --cfg=configs/nuscenes/usa_singapore/xmuda_pl.yaml
```

You can start the trainings on the other UDA scenarios (Day/Night and A2D2/SemanticKITTI) analogously:
```
$ python xmuda/test.py --cfg=configs/nuscenes/day_night/xmuda.yaml --pselab @/model_2d_100000.pth @/model_3d_100000.pth DATASET_TARGET.TEST "('train_night',)"
$ python xmuda/train_xmuda.py --cfg=configs/nuscenes/day_night/xmuda_pl.yaml

# use batch size 1, because of different image sizes Kitti
$ python xmuda/test.py --cfg=configs/a2d2_semantic_kitti/xmuda.yaml --pselab @/model_2d_100000.pth @/model_3d_100000.pth DATASET_TARGET.TEST "('train',)" VAL.BATCH_SIZE 1
$ python xmuda/train_xmuda.py --cfg=configs/a2d2_semantic_kitti/xmuda_pl.yaml
```

### Baseline
Train the baselines (only on source) with:
```
$ python xmuda/train_baseline.py --cfg=configs/nuscenes/usa_singapore/baseline.yaml
$ python xmuda/train_baseline.py --cfg=configs/nuscenes/day_night/baseline.yaml
$ python xmuda/train_baseline.py --cfg=configs/a2d2_semantic_kitti/baseline.yaml
```

## Testing
You can provide which checkpoints you want to use for testing. We used the ones
that performed best on the validation set during training (the best val iteration for 2D and 3D is
shown at the end of each training). Note that `@` will be replaced
by the output directory for that config file. For example:
```
$ cd <root dir of this repo>
$ python xmuda/test.py --cfg=configs/nuscenes/usa_singapore/xmuda.yaml @/model_2d_065000.pth @/model_3d_095000.pth
```
You can also provide an absolute path without `@`. 

## Model Zoo

You can download the models with the scores below from
[this Google drive folder](https://drive.google.com/drive/folders/16MTKz4LOIwqQc3Vo6LAGrpiIC72hvggc?usp=sharing).

| Method | USA/Singapore 2D | USA/Singapore 3D | Day/Night 2D | Day/Night 3D | A2D2/Sem.KITTI 2D | A2D2/Sem.KITTI 3D |
| --- | --- | --- | --- | --- | --- |  --- | 
| Baseline (source only)  | 53.4 | 46.5 | 42.2 | 41.2 | 34.2<sup>*</sup> | 35.9<sup>*</sup>
| xMUDA  | 59.3 | 52.0 | 46.2 | 44.2 | 38.3<sup>*</sup> | 46.0<sup>*</sup>
| xMUDA<sub>PL</sub> |61.1 | 54.1 | 47.1 | 46.7 | 41.2<sup>*</sup> | 49.8<sup>*</sup>

<sup>*</sup> Slight differences from the paper on A2D2/Sem.KITTI: Now we use class weights computed on source.
In the paper, we falsely computed class weights on the target domain.

## Acknowledgements
Note that this code borrows from the [MVPNet](https://github.com/maxjaritz/mvpnet) repo.

## License
xMUDA is released under the [Apache 2.0 license](./LICENSE).
