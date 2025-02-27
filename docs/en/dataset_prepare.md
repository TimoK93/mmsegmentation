<!-- #region -->

## Prepare datasets

It is recommended to symlink the dataset root to `$MMSEGMENTATION/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.

```none
mmsegmentation
├── mmseg
├── tools
├── configs
├── data
│   ├── cityscapes
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   ├── val
│   ├── VOCdevkit
│   │   ├── VOC2012
│   │   │   ├── JPEGImages
│   │   │   ├── SegmentationClass
│   │   │   ├── ImageSets
│   │   │   │   ├── Segmentation
│   │   ├── VOC2010
│   │   │   ├── JPEGImages
│   │   │   ├── SegmentationClassContext
│   │   │   ├── ImageSets
│   │   │   │   ├── SegmentationContext
│   │   │   │   │   ├── train.txt
│   │   │   │   │   ├── val.txt
│   │   │   ├── trainval_merged.json
│   │   ├── VOCaug
│   │   │   ├── dataset
│   │   │   │   ├── cls
│   ├── ade
│   │   ├── ADEChallengeData2016
│   │   │   ├── annotations
│   │   │   │   ├── training
│   │   │   │   ├── validation
│   │   │   ├── images
│   │   │   │   ├── training
│   │   │   │   ├── validation
│   ├── coco_stuff10k
│   │   ├── images
│   │   │   ├── train2014
│   │   │   ├── test2014
│   │   ├── annotations
│   │   │   ├── train2014
│   │   │   ├── test2014
│   │   ├── imagesLists
│   │   │   ├── train.txt
│   │   │   ├── test.txt
│   │   │   ├── all.txt
│   ├── coco_stuff164k
│   │   ├── images
│   │   │   ├── train2017
│   │   │   ├── val2017
│   │   ├── annotations
│   │   │   ├── train2017
│   │   │   ├── val2017
│   ├── CHASE_DB1
│   │   ├── images
│   │   │   ├── training
│   │   │   ├── validation
│   │   ├── annotations
│   │   │   ├── training
│   │   │   ├── validation
│   ├── DRIVE
│   │   ├── images
│   │   │   ├── training
│   │   │   ├── validation
│   │   ├── annotations
│   │   │   ├── training
│   │   │   ├── validation
│   ├── HRF
│   │   ├── images
│   │   │   ├── training
│   │   │   ├── validation
│   │   ├── annotations
│   │   │   ├── training
│   │   │   ├── validation
│   ├── STARE
│   │   ├── images
│   │   │   ├── training
│   │   │   ├── validation
│   │   ├── annotations
│   │   │   ├── training
│   │   │   ├── validation
|   ├── dark_zurich
|   │   ├── gps
|   │   │   ├── val
|   │   │   └── val_ref
|   │   ├── gt
|   │   │   └── val
|   │   ├── LICENSE.txt
|   │   ├── lists_file_names
|   │   │   ├── val_filenames.txt
|   │   │   └── val_ref_filenames.txt
|   │   ├── README.md
|   │   └── rgb_anon
|   │   |   ├── val
|   │   |   └── val_ref
|   ├── NighttimeDrivingTest
|   |   ├── gtCoarse_daytime_trainvaltest
|   |   │   └── test
|   |   │       └── night
|   |   └── leftImg8bit
|   |   |   └── test
|   |   |       └── night
│   ├── loveDA
│   │   ├── img_dir
│   │   │   ├── train
│   │   │   ├── val
│   │   │   ├── test
│   │   ├── ann_dir
│   │   │   ├── train
│   │   │   ├── val
│   ├── potsdam
│   │   ├── img_dir
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── ann_dir
│   │   │   ├── train
│   │   │   ├── val
│   ├── vaihingen
│   │   ├── img_dir
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── ann_dir
│   │   │   ├── train
│   │   │   ├── val
│   ├── iSAID
│   │   ├── img_dir
│   │   │   ├── train
│   │   │   ├── val
│   │   │   ├── test
│   │   ├── ann_dir
│   │   │   ├── train
│   │   │   ├── val
│   ├── occlusion-aware-face-dataset
│   │   ├── train.txt
│   │   ├── NatOcc_hand_sot
│   │   │   ├── img
│   │   │   ├── mask
│   │   ├── NatOcc_object
│   │   │   ├── img
│   │   │   ├── mask
│   │   ├── RandOcc
│   │   │   ├── img
│   │   │   ├── mask
│   │   ├── RealOcc
│   │   │   ├── img
│   │   │   ├── mask
│   │   │   ├── split
│   ├── ImageNetS
│   │   ├── ImageNetS919
│   │   │   ├── train-semi
│   │   │   ├── train-semi-segmentation
│   │   │   ├── validation
│   │   │   ├── validation-segmentation
│   │   │   ├── test
│   │   ├── ImageNetS300
│   │   │   ├── train-semi
│   │   │   ├── train-semi-segmentation
│   │   │   ├── validation
│   │   │   ├── validation-segmentation
│   │   │   ├── test
│   │   ├── ImageNetS50
│   │   │   ├── train-semi
│   │   │   ├── train-semi-segmentation
│   │   │   ├── validation
│   │   │   ├── validation-segmentation
│   │   │   ├── test
│   ├── kitti_step
│   │   ├── testing
│   │   ├── training
│   │   ├── panoptic_maps
```

### Cityscapes

The data could be found [here](https://www.cityscapes-dataset.com/downloads/) after registration.

By convention, `**labelTrainIds.png` are used for cityscapes training.
We provided a [scripts](https://github.com/open-mmlab/mmsegmentation/blob/master/tools/convert_datasets/cityscapes.py) based on [cityscapesscripts](https://github.com/mcordts/cityscapesScripts)
to generate `**labelTrainIds.png`.

```shell
# --nproc means 8 process for conversion, which could be omitted as well.
python tools/convert_datasets/cityscapes.py data/cityscapes --nproc 8
```

### Pascal VOC

Pascal VOC 2012 could be downloaded from [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar).
Beside, most recent works on Pascal VOC dataset usually exploit extra augmentation data, which could be found [here](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz).

If you would like to use augmented VOC dataset, please run following command to convert augmentation annotations into proper format.

```shell
# --nproc means 8 process for conversion, which could be omitted as well.
python tools/convert_datasets/voc_aug.py data/VOCdevkit data/VOCdevkit/VOCaug --nproc 8
```

Please refer to [concat dataset](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/tutorials/customize_datasets.md#concatenate-dataset) for details about how to concatenate them and train them together.

### ADE20K

The training and validation set of ADE20K could be download from this [link](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip).
We may also download test set from [here](http://data.csail.mit.edu/places/ADEchallenge/release_test.zip).

### Pascal Context

The training and validation set of Pascal Context could be download from [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar). You may also download test set from [here](http://host.robots.ox.ac.uk:8080/eval/downloads/VOC2010test.tar) after registration.

To split the training and validation set from original dataset, you may download trainval_merged.json from [here](https://codalabuser.blob.core.windows.net/public/trainval_merged.json).

If you would like to use Pascal Context dataset, please install [Detail](https://github.com/zhanghang1989/detail-api) and then run the following command to convert annotations into proper format.

```shell
python tools/convert_datasets/pascal_context.py data/VOCdevkit data/VOCdevkit/VOC2010/trainval_merged.json
```

### COCO Stuff 10k

The data could be downloaded [here](http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/cocostuff-10k-v1.1.zip) by wget.

For COCO Stuff 10k dataset, please run the following commands to download and convert the dataset.

```shell
# download
mkdir coco_stuff10k && cd coco_stuff10k
wget http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/cocostuff-10k-v1.1.zip

# unzip
unzip cocostuff-10k-v1.1.zip

# --nproc means 8 process for conversion, which could be omitted as well.
python tools/convert_datasets/coco_stuff10k.py /path/to/coco_stuff10k --nproc 8
```

By convention, mask labels in `/path/to/coco_stuff164k/annotations/*2014/*_labelTrainIds.png` are used for COCO Stuff 10k training and testing.

### COCO Stuff 164k

For COCO Stuff 164k dataset, please run the following commands to download and convert the augmented dataset.

```shell
# download
mkdir coco_stuff164k && cd coco_stuff164k
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip

# unzip
unzip train2017.zip -d images/
unzip val2017.zip -d images/
unzip stuffthingmaps_trainval2017.zip -d annotations/

# --nproc means 8 process for conversion, which could be omitted as well.
python tools/convert_datasets/coco_stuff164k.py /path/to/coco_stuff164k --nproc 8
```

By convention, mask labels in `/path/to/coco_stuff164k/annotations/*2017/*_labelTrainIds.png` are used for COCO Stuff 164k training and testing.

The details of this dataset could be found at [here](https://github.com/nightrome/cocostuff#downloads).

### CHASE DB1

The training and validation set of CHASE DB1 could be download from [here](https://staffnet.kingston.ac.uk/~ku15565/CHASE_DB1/assets/CHASEDB1.zip).

To convert CHASE DB1 dataset to MMSegmentation format, you should run the following command:

```shell
python tools/convert_datasets/chase_db1.py /path/to/CHASEDB1.zip
```

The script will make directory structure automatically.

### DRIVE

The training and validation set of DRIVE could be download from [here](https://drive.grand-challenge.org/). Before that, you should register an account. Currently '1st_manual' is not provided officially.

To convert DRIVE dataset to MMSegmentation format, you should run the following command:

```shell
python tools/convert_datasets/drive.py /path/to/training.zip /path/to/test.zip
```

The script will make directory structure automatically.

### HRF

First, download [healthy.zip](https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/healthy.zip), [glaucoma.zip](https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/glaucoma.zip), [diabetic_retinopathy.zip](https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/diabetic_retinopathy.zip), [healthy_manualsegm.zip](https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/healthy_manualsegm.zip), [glaucoma_manualsegm.zip](https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/glaucoma_manualsegm.zip) and [diabetic_retinopathy_manualsegm.zip](https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/diabetic_retinopathy_manualsegm.zip).

To convert HRF dataset to MMSegmentation format, you should run the following command:

```shell
python tools/convert_datasets/hrf.py /path/to/healthy.zip /path/to/healthy_manualsegm.zip /path/to/glaucoma.zip /path/to/glaucoma_manualsegm.zip /path/to/diabetic_retinopathy.zip /path/to/diabetic_retinopathy_manualsegm.zip
```

The script will make directory structure automatically.

### STARE

First, download [stare-images.tar](http://cecas.clemson.edu/~ahoover/stare/probing/stare-images.tar), [labels-ah.tar](http://cecas.clemson.edu/~ahoover/stare/probing/labels-ah.tar) and [labels-vk.tar](http://cecas.clemson.edu/~ahoover/stare/probing/labels-vk.tar).

To convert STARE dataset to MMSegmentation format, you should run the following command:

```shell
python tools/convert_datasets/stare.py /path/to/stare-images.tar /path/to/labels-ah.tar /path/to/labels-vk.tar
```

The script will make directory structure automatically.

### Dark Zurich

Since we only support test models on this dataset, you may only download [the validation set](https://data.vision.ee.ethz.ch/csakarid/shared/GCMA_UIoU/Dark_Zurich_val_anon.zip).

### Nighttime Driving

Since we only support test models on this dataset, you may only download [the test set](http://data.vision.ee.ethz.ch/daid/NighttimeDriving/NighttimeDrivingTest.zip).

### LoveDA

The data could be downloaded from Google Drive [here](https://drive.google.com/drive/folders/1ibYV0qwn4yuuh068Rnc-w4tPi0U0c-ti?usp=sharing).

Or it can be downloaded from [zenodo](https://zenodo.org/record/5706578#.YZvN7SYRXdF), you should run the following command:

```shell
# Download Train.zip
wget https://zenodo.org/record/5706578/files/Train.zip
# Download Val.zip
wget https://zenodo.org/record/5706578/files/Val.zip
# Download Test.zip
wget https://zenodo.org/record/5706578/files/Test.zip
```

For LoveDA dataset, please run the following command to download and re-organize the dataset.

```shell
python tools/convert_datasets/loveda.py /path/to/loveDA
```

Using trained model to predict test set of LoveDA and submit it to server can be found [here](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/inference.md).

More details about LoveDA can be found [here](https://github.com/Junjue-Wang/LoveDA).

### ISPRS Potsdam

The [Potsdam](https://www2.isprs.org/commissions/comm2/wg4/benchmark/2d-sem-label-potsdam/)
dataset is for urban semantic segmentation used in the 2D Semantic Labeling Contest - Potsdam.

The dataset can be requested at the challenge [homepage](https://www2.isprs.org/commissions/comm2/wg4/benchmark/data-request-form/).
The '2_Ortho_RGB.zip' and '5_Labels_all_noBoundary.zip' are required.

For Potsdam dataset, please run the following command to download and re-organize the dataset.

```shell
python tools/convert_datasets/potsdam.py /path/to/potsdam
```

In our default setting, it will generate 3456 images for training and 2016 images for validation.

### ISPRS Vaihingen

The [Vaihingen](https://www2.isprs.org/commissions/comm2/wg4/benchmark/2d-sem-label-vaihingen/)
dataset is for urban semantic segmentation used in the 2D Semantic Labeling Contest - Vaihingen.

The dataset can be requested at the challenge [homepage](https://www2.isprs.org/commissions/comm2/wg4/benchmark/data-request-form/).
The 'ISPRS_semantic_labeling_Vaihingen.zip' and 'ISPRS_semantic_labeling_Vaihingen_ground_truth_eroded_COMPLETE.zip' are required.

For Vaihingen dataset, please run the following command to download and re-organize the dataset.

```shell
python tools/convert_datasets/vaihingen.py /path/to/vaihingen
```

In our default setting (`clip_size` =512, `stride_size`=256), it will generate 344 images for training and 398 images for validation.

### iSAID

The data images could be download from [DOTA-v1.0](https://captain-whu.github.io/DOTA/dataset.html) (train/val/test)

The data annotations could be download from [iSAID](https://captain-whu.github.io/iSAID/dataset.html) (train/val)

The dataset is a Large-scale Dataset for Instance Segmentation (also have segmantic segmentation) in Aerial Images.

You may need to follow the following structure for dataset preparation after downloading iSAID dataset.

```
│   ├── iSAID
│   │   ├── train
│   │   │   ├── images
│   │   │   │   ├── part1.zip
│   │   │   │   ├── part2.zip
│   │   │   │   ├── part3.zip
│   │   │   ├── Semantic_masks
│   │   │   │   ├── images.zip
│   │   ├── val
│   │   │   ├── images
│   │   │   │   ├── part1.zip
│   │   │   ├── Semantic_masks
│   │   │   │   ├── images.zip
│   │   ├── test
│   │   │   ├── images
│   │   │   │   ├── part1.zip
│   │   │   │   ├── part2.zip
```

```shell
python tools/convert_datasets/isaid.py /path/to/iSAID
```

In our default setting (`patch_width`=896, `patch_height`=896,　`overlap_area`=384), it will generate 33978 images for training and 11644 images for validation.

### Delving into High-Quality Synthetic Face Occlusion Segmentation Datasets

The dataset is generated by two techniques, Naturalistic occlusion generation, Random occlusion generation. you must install face-occlusion-generation and dataset. see more guide in https://github.com/kennyvoo/face-occlusion-generation.git

### KITTI-STEP

After registration, the data images could be download from [KITTI-STEP](http://www.cvlibs.net/datasets/kitti/eval_step.php)

You may need to follow the following structure for dataset preparation after downloading iSAID dataset.

```
│   ├── kitti_step
│   │   ├── testing
│   │   ├── training
│   │   ├── panoptic_maps
```

Run the preparation script to generate label files and kitti subsets by executing

```shell
python tools/convert_datasets/kitti_step.py /path/to/kitti_step
```

After executing the script, your directory should look like

```
│   ├── kitti_step
│   │   ├── testing
│   │   ├── training
│   │   ├── panoptic_maps
│   │   ├── training_openmmlab
│   │   ├── panoptic_maps_openmmlab
```

## Dataset Preparation

step 1

Create a folder for data generation materials on mmsegmentation folder.

```shell
mkdir data_materials
```

step 2

Please download the masks (11k-hands_mask.7z,CelebAMask-HQ-masks_corrected.7z) from this [drive](https://drive.google.com/drive/folders/15nZETWlGMdcKY6aHbchRsWkUI42KTNs5?usp=sharing)

Please download the images from [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ), [11k Hands.zip](https://sites.google.com/view/11khands) and [dtd-r1.0.1.tar.gz](https://www.robots.ox.ac.uk/~vgg/data/dtd/).

step 3

Download a upsampled COCO objects images and masks (coco_object.7z). files can be found in this [drive](https://drive.google.com/drive/folders/15nZETWlGMdcKY6aHbchRsWkUI42KTNs5?usp=sharing).

Download CelebAMask-HQ and 11k Hands images split txt files. (11k_hands_sample.txt, CelebAMask-HQ-WO-train.txt) found in [drive](https://drive.google.com/drive/folders/15nZETWlGMdcKY6aHbchRsWkUI42KTNs5?usp=sharing).

download file to ./data_materials

```none
CelebAMask-HQ.zip
CelebAMask-HQ-masks_corrected.7z
CelebAMask-HQ-WO-train.txt
RealOcc.7z
RealOcc-Wild.7z
11k-hands_mask.7z
11k Hands.zip
11k_hands_sample.txt
coco_object.7z
dtd-r1.0.1.tar.gz
```

______________________________________________________________________

```bash
apt-get install p7zip-full

cd data_materials

#make occlusion-aware-face-dataset folder
mkdir path-to-mmsegmentaion/data/occlusion-aware-face-dataset

#extract celebAMask-HQ and split by train-set
unzip CelebAMask-HQ.zip
7za x CelebAMask-HQ-masks_corrected.7z -o./CelebAMask-HQ
#copy training data to train-image-folder
rsync -a ./CelebAMask-HQ/CelebA-HQ-img/ --files-from=./CelebAMask-HQ-WO-train.txt ./CelebAMask-HQ-WO-Train_img
#create a file-name txt file for copying mask
basename -s .jpg ./CelebAMask-HQ-WO-Train_img/* > train.txt
#add .png to file-name txt file
xargs -n 1 -i echo {}.png < train.txt > mask_train.txt
#copy training data to train-mask-folder
rsync -a ./CelebAMask-HQ/CelebAMask-HQ-masks_corrected/ --files-from=./mask_train.txt ./CelebAMask-HQ-WO-Train_mask
mv train.txt ../data/occlusion-aware-face-dataset

#extract DTD
tar -zxvf dtd-r1.0.1.tar.gz
mv dtd DTD

#extract hands dataset and split by 200 samples
7za x 11k-hands_masks.7z -o.
unzip Hands.zip
rsync -a ./Hands/ --files-from=./11k_hands_sample.txt ./11k-hands_img

#extract upscaled coco object
7za x coco_object.7z -o.
mv coco_object/* .

#extract validation set
7za x RealOcc.7z -o../data/occlusion-aware-face-dataset

```

**Dataset material Organization:**

```none

├── data_materials
│   ├── CelebAMask-HQ-WO-Train_img
│   │   ├── {image}.jpg
│   ├── CelebAMask-HQ-WO-Train_mask
│   │   ├── {mask}.png
│   ├── DTD
│   │   ├── images
│   │   │   ├── {classA}
│   │   │   │   ├── {image}.jpg
│   │   │   ├── {classB}
│   │   │   │   ├── {image}.jpg
│   ├── 11k-hands_img
│   │   ├── {image}.jpg
│   ├── 11k-hands_mask
│   │   ├── {mask}.png
│   ├── object_image_sr
│   │   ├── {image}.jpg
│   ├── object_mask_x4
│   │   ├── {mask}.png

```

## Data Generation

```bash
git clone https://github.com/kennyvoo/face-occlusion-generation.git
cd face_occlusion-generation
```

Example script to generate NatOcc hand dataset

```bash
CUDA_VISIBLE_DEVICES=0 NUM_WORKERS=4 python main.py \
--config ./configs/natocc_hand.yaml \
--opts OUTPUT_PATH "path/to/mmsegmentation/data/occlusion-aware-face-dataset/NatOcc_hand_sot"\
AUGMENTATION.SOT True \
SOURCE_DATASET.IMG_DIR "path/to/data_materials/CelebAMask-HQ-WO-Train_img" \
SOURCE_DATASET.MASK_DIR "path/to/mmsegmentation/data_materials/CelebAMask-HQ-WO-Train_mask" \
OCCLUDER_DATASET.IMG_DIR "path/to/mmsegmentation/data_materials/11k-hands_img" \
OCCLUDER_DATASET.MASK_DIR "path/to/mmsegmentation/data_materials/11k-hands_masks"
```

Example script to generate NatOcc object dataset

```bash
CUDA_VISIBLE_DEVICES=0 NUM_WORKERS=4 python main.py \
--config ./configs/natocc_objects.yaml \
--opts OUTPUT_PATH "path/to/mmsegmentation/data/occlusion-aware-face-dataset/NatOcc_object" \
SOURCE_DATASET.IMG_DIR "path/to/mmsegmentation/data_materials/CelebAMask-HQ-WO-Train_img" \
SOURCE_DATASET.MASK_DIR "path/to/mmsegmentation/data_materials/CelebAMask-HQ-WO-Train_mask" \
OCCLUDER_DATASET.IMG_DIR "path/to/mmsegmentation/data_materials/object_image_sr" \
OCCLUDER_DATASET.MASK_DIR "path/to/mmsegmentation/data_materials/object_mask_x4"
```

Example script to generate RandOcc dataset

```bash
CUDA_VISIBLE_DEVICES=0 NUM_WORKERS=4  python main.py \
--config ./configs/randocc.yaml \
--opts OUTPUT_PATH "path/to/mmsegmentation/data/occlusion-aware-face-dataset/RandOcc" \
SOURCE_DATASET.IMG_DIR "path/to/mmsegmentation/data_materials/CelebAMask-HQ-WO-Train_img/" \
SOURCE_DATASET.MASK_DIR "path/to/mmsegmentation/data_materials/CelebAMask-HQ-WO-Train_mask" \
OCCLUDER_DATASET.IMG_DIR "path/to/jw93/mmsegmentation/data_materials/DTD/images"
```

**Dataset Organization:**

```none
├── data
│   ├── occlusion-aware-face-dataset
│   │   ├── train.txt
│   │   ├── NatOcc_hand_sot
│   │   │   ├── img
│   │   │   │   ├── {image}.jpg
│   │   │   ├── mask
│   │   │   │   ├── {mask}.png
│   │   ├── NatOcc_object
│   │   │   ├── img
│   │   │   │   ├── {image}.jpg
│   │   │   ├── mask
│   │   │   │   ├── {mask}.png
│   │   ├── RandOcc
│   │   │   ├── img
│   │   │   │   ├── {image}.jpg
│   │   │   ├── mask
│   │   │   │   ├── {mask}.png
│   │   ├── RealOcc
│   │   │   ├── img
│   │   │   │   ├── {image}.jpg
│   │   │   ├── mask
│   │   │   │   ├── {mask}.png
│   │   │   ├── split
│   │   │   │   ├── val.txt
```

<!-- #endregion -->

```python

```

### ImageNetS

The ImageNet-S dataset is for [Large-scale unsupervised/semi-supervised semantic segmentation](https://arxiv.org/abs/2106.03149).

The images and annotations are available on [ImageNet-S](https://github.com/LUSSeg/ImageNet-S#imagenet-s-dataset-preparation).

```
│   ├── ImageNetS
│   │   ├── ImageNetS919
│   │   │   ├── train-semi
│   │   │   ├── train-semi-segmentation
│   │   │   ├── validation
│   │   │   ├── validation-segmentation
│   │   │   ├── test
│   │   ├── ImageNetS300
│   │   │   ├── train-semi
│   │   │   ├── train-semi-segmentation
│   │   │   ├── validation
│   │   │   ├── validation-segmentation
│   │   │   ├── test
│   │   ├── ImageNetS50
│   │   │   ├── train-semi
│   │   │   ├── train-semi-segmentation
│   │   │   ├── validation
│   │   │   ├── validation-segmentation
│   │   │   ├── test
```
