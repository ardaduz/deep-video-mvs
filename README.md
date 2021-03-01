# deep-video-mvs
## DeepVideoMVS: Multi-View Stereo on Video with Recurrent Spatio-Temporal Fusion
![](miscellaneous/teaser.png)

### Paper (Accepted to CVPR 2021): [arXiv](https://arxiv.org/abs/2012.02177)

### Introduction
DeepVideoMVS is a learning-based online multi-view depth prediction approach on 
posed video streams, where the scene geometry information computed in the previous 
time steps is propagated to the current time step. The backbone of the approach is a 
real-time capable, lightweight encoder-decoder that relies on cost volumes computed 
from pairs of images.  We extend it with a ConvLSTM cell at the bottleneck layer, 
and a hidden state propagation scheme where we partially account for the viewpoint 
changes between time steps.

![](miscellaneous/architecture.png)

This extension brings only a small overhead of computation time and memory consumption over the
backbone, while improving the depth predictions significantly. The fusion network produces noticeably 
more consistent depth predictions for the planar surfaces throughout a sequence which gets reflected 
as smooth reconstructions of such surfaces. This is demonstrated in the
[video](miscellaneous/deep-video-mvs-supplementary-video.mp4).

If you find this project useful for your research, please cite:
```
@misc{düzçeker2020deepvideomvs,
      title={DeepVideoMVS: Multi-View Stereo on Video with Recurrent Spatio-Temporal Fusion}, 
      author={Arda Düzçeker and Silvano Galliani and Christoph Vogel and Pablo Speciale and Mihai Dusmanu and Marc Pollefeys},
      year={2020},
      eprint={2012.02177},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

### Dependencies / Installation
```
conda create -n dvmvs-env
conda activate dvmvs-env
conda install -y -c conda-forge -c pytorch -c fvcore -c pytorch3d python=3.8 pytorch=1.5.1 torchvision=0.6.1 cudatoolkit=10.2 opencv=4.4.0 tqdm=4.50.2 scipy=1.5.2 fvcore=0.1.2 pytorch3d=0.2.5
pip install pytools==2020.4 kornia==0.3.2 path==15.0.0 protobuf==3.13.0 tensorboardx==2.1

git clone https://github.com/ardaduz/deep-video-mvs.git
pip install -e deep-video-mvs
```

### Data Structure
The scripts for parsing the datasets are provided in the [dataset](https://github.com/ardaduz/deep-video-mvs/tree/master/dataset) folder.
All of the scripts might not work straight-ahead due to naming and foldering conventions while
downloading the datasets, however they should help reduce the effort required. Exporting [ScanNet](http://www.scan-net.org/) .sens 
files, both for training and testing, should work with very minimal effort. The script that is provided here is
a modified version of the official code and similarly requires python2.

During testing, the system expects a data structure for a particular scene as provided in the 
sample-data/hololens-dataset/000. We assume PNG format for all images.
* "images" folder contains the input images that will be used by the model 
and the naming convention is not important. The system considers the sequential order
alphabetically.
* "depth" folder contains the groundtruth depth maps that are used for metric evaluation,
the names must match with the color images. The depth images must be uint16 PNG format, 
and the depth value in millimeters. 
For example, if the depth is 1.12 meters for a pixel location,
it should read 1120 in the groundtruth depth image.
* "poses.txt" contains CAMERA-TO-WORLD pose corresponding to each color and depth image. 
Each line is one flattened pose in homogeneous coordinates.
* "K.txt" is the intrinsic matrix for a given sequence after the images are undistorted.

During training, the system expects each scene to be placed in a folder, and color image and depth image for a time step 
to be packed inside a zipped numpy archive (.npz). See the [code](https://github.com/ardaduz/deep-video-mvs/blob/master/dataset/scannet-export/scannet-export.py#L151).
We use frame_skip=4 while exporting the ScanNet training and validation scenes due to the large amount of data.
The training/validation split of unique scenes which are used during this work is also provided 
[here](https://github.com/ardaduz/deep-video-mvs/blob/master/dataset/scannet-export),
one may replace the randomly generated ones with these two.

### Training / Testing
There are no command line arguments for the system.
Instead, many general parameters are controlled from the config.py.
Please adjust the folder locations (and/or other settings) inside the config.py.

Additionally, several important training hyperparameters are controlled directly
inside the training scripts in a [class](https://github.com/ardaduz/deep-video-mvs/blob/master/dvmvs/fusionnet/run-training.py#L18).

#### Training:
To train the networks from scratch, please refer to the detailed explanation
of the procedure that we follow provided in the supplementary of the paper.
In summary, we first train the pairnet independently and use some modules' weights
to partially initialize our fusionnet. For fusionnet, we start by training the cell 
and the decoder, which are randomly initialized, and then gradually unfreeze the 
other modules. Finally, we finetune only the cell while warping the hidden states with
the predictions instead of the groundtruth depths.
* pairnet training script:
    ```
    cd deep-video-mvs/dvmvs/pairnet
    python run-training.py
    ```
* fusionnet training script:
    ```
    cd deep-video-mvs/dvmvs/fusionnet
    python run-training.py
    ```
    
#### Testing:
There are two types of test runs one can perform:
 
First is run-testing.py and it is for evaluating on multiple datasets and/or sequences 
at a run. This script requires pre-selected keyframe filenames for the desired sequences
similar to the ones provided in the sample-data/indices.
One can determine custom keyframe filenames with custom keyframe selection approaches,
or we provide a simple simulation of our keyframe selection heuristic in
simulate_keyframe_buffer.py. The results are saved to the Config.test_result_folder.

Second is to run the testing in a more online fashion. One can specify a single scene
inside the Config, and use run-testing-online.py to evaluate on this particular scene. 
In this script, we use our keyframe selection heuristic on-the-go and predict the depth maps 
for the selected keyframes (Attention! We do not predict depth maps for all images). 
The results are saved to the current directory.
To run the online testing:
```
cd deep-video-mvs/dvmvs/fusionnet
python run-testing-online.py
``` 

Predicted depth maps for a scene and
average error of each frame are saved in .npz format. 
Errors contain 8 different metrics for each frame in order:
abs_error, abs_relative_error, abs_inverse_error, squared_relative_error, 
rmse, ratio_125, ratio_125_2, ratio_125_3. They can be accessed with:
```
predictions = numpy.load(prediction_filename)['arr_0']
errors = numpy.load(error_filename)['arr_0']
```
