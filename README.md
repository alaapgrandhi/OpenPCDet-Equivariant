## Installation instructions
### Getting the code
When cloning this codebase, the submodules need to be cloned as well to ensure that the repository builds correctly. This can be done by running the following command:
```
git clone --recurse-submodules git@github.com:alaapgrandhi/OpenPCDet-Equivariant.git
```

### Setting up the environment
- Enter the base code folder by running `cd OpenPCDet-Equivariant`
- Make the scripts executable by running `chmod +x ./setup.sh ./get_models.sh`
- Create the conda environment and build the library by running `./setup.sh`
- Download the necessary pretrained model checkpoints by running `./get_models.sh` 

### Using the environment
Now that the environment has been set up, you can simply start it by running `conda activate ESF` and stop it by running `conda deactivate ESF`. Any scripts and Python files in this repository should only be run within the ESF environment to isolate dependency troubles to that environment.

### Downloading the NuScenes dataset
When browsing the downloads section on [nuScenes' website](https://www.nuscenes.org/nuscenes#download), you should use the download links under the Full Dataset (v1.0) heading. Under this heading, three different versions of the dataset appear (Mini, Trainval, and Test). If disk space is not an issue, I recommend downloading and unpacking the complete Trainval set. Otherwise, the Mini version of the dataset can also work for simple visualization/testing (but it is not recommended for training). The Test version is not recommended here as it is unannotated and cannot be used for evaluation purposes.

### Setting up the NuScenes dataset
OpenPCDet's [dataset preparation guide](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md) can be used to set up the NuScenes dataset, but you should skip the `pip install nuscenes-devkit==1.0.5` step as it can break the dependencies chain.

## Running training
Model training can be run by cd'ing into the tools subfolder and then running the following command:
```
python train.py --cfg_file ${CONFIG_FILE}
```

If you wish to train the base BEVFusion model on the NuScenes dataset (if you downloaded the trainval set), replace `${CONFIG_FILE}` with `cfgs/nuscenes_models/bevfusion.yaml`. 

## Running inference
Model inference can be run by cd'ing into the tools subfolder and then running the following command:
```
python infer.py --cfg_file ${CONFIG_FILE} --ckpt ${PATH_TO_CHECKPOINT} --data_path ${PATH_TO_DATA} --pc_fmt ${LIDAR_FORMAT} --im_fmt ${IMAGE_FORMAT} --thresh ${THRESHOLD}
```
The various flags here have the following impacts and suggested values:
- `${CONFIG_FILE}` represents the path to the model config, and can be replaced with `cfgs/nuscenes_models/bevfusion.yaml`
- `${PATH_TO_CHECKPOINT}` represents the path to the pretrained model checkpoint and can be replaced with `../pretrained_models/nuscenes_bevfusion.pth`
- `${PATH_TO_DATA}` represents the path to the data on which to infer, and can be replaced with `sample_data/sample_data_cam_jpg` (The data folders in sample_data can be inspected to see the expected data file hierarchy/structure)
- `${LIDAR_FORMAT}` represents the file format used for the LiDAR data, is expected to be one of [pcd, np, pt], and can be replaced with `pcd`
- `${IMAGE_FORMAT}` represents the file format used for the Camera data, is expected to be one of [png, jpg, np, pt], and can be replaced with `jpg`
- `${THRESHOLD}` represents a float filtering threshold used to ignore unconfident predicted bounding boxes, and can be replaced with `0.05`

## Running evaluation
Model Evaluation can be run by cd'ing into the tools subfolder and then running the following command:
```
python test.py --cfg_file ${CONFIG_FILE} --ckpt ${PATH_TO_CHECKPOINT} --batch_size ${BATCH_SIZE}
```

If you wish to evaluate the base BEVFusion model checkpoint provided by OpenPCDet, replace `${CONFIG_FILE}` with `cfgs/nuscenes_models/bevfusion.yaml`, replace `${PATH_TO_CHECKPOINT}` with `../pretrained_models/nuscenes_bevfusion.pth`, and replace `${BATCH_SIZE}` with `1` (to ensure minimal VRAM requirement).

## Running unit tests
Unit tests can be run by cd'ing into the tests subfolder and then running the following command (numerous warnings are thrown by the base OpenPCDet code and as such they are disabled to prevent cluttering the pytest output):
```
pytest --disable-warnings
```

## Module relations with the code

