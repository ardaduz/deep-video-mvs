## *DeepVideoMVS*: Multi-View Stereo on Video with Recurrent Spatio-Temporal Fusion
### Paper (CVPR 2021): [arXiv](https://arxiv.org/abs/2012.02177)
### Presentation: *Coming Soon*

https://user-images.githubusercontent.com/46934354/122122414-0fde2900-ce2d-11eb-9276-16d362438249.mp4

<br />

***DeepVideoMVS*** is a learning-based online multi-view depth prediction approach on 
posed video streams, where the scene geometry information computed in the previous 
time steps is propagated to the current time step. The backbone of the approach is a 
real-time capable, lightweight encoder-decoder that relies on cost volumes computed 
from pairs of images.  We extend it with a ConvLSTM cell at the bottleneck layer, 
and a hidden state propagation scheme where we partially account for the viewpoint 
changes between time steps.

This extension brings only a small overhead of computation time and memory consumption over the
backbone, while improving the depth predictions significantly. As the result, DeepVideoMVS achieves
**highly accurate depth maps** with **real-time performance** and **low memory consumption**. 
It produces noticeably more consistent depth predictions than our backbone and the existing 
methods throughout a sequence, which gets reflected as less noisy reconstructions.

<br />

https://user-images.githubusercontent.com/46934354/122127322-4c148800-ce33-11eb-982a-4ccf71a7c54a.mp4

<br />

![](miscellaneous/teaser.jpg)

---
### Citation
---
If you find this project useful for your research, please cite:
```
@inproceedings{Duzceker_2021_CVPR,
    author    = {Duzceker, Arda and Galliani, Silvano and Vogel, Christoph and Speciale, Pablo and Dusmanu, Mihai and Pollefeys, Marc},
    title     = {DeepVideoMVS: Multi-View Stereo on Video With Recurrent Spatio-Temporal Fusion},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {15324-15333}
}
```

<br />

---
### Dependencies / Installation
---
```
conda create -n dvmvs-env
conda activate dvmvs-env
conda install -y -c conda-forge -c pytorch -c fvcore -c pytorch3d \
    python=3.8 \
    pytorch=1.5.1 \
    torchvision=0.6.1 \
    cudatoolkit=10.2 \
    opencv=4.4.0 \
    tqdm=4.50.2 \
    scipy=1.5.2 \
    fvcore=0.1.2 \
    pytorch3d=0.2.5
pip install \
    pytools==2020.4 \
    kornia==0.3.2 \
    path==15.0.0 \
    protobuf==3.13.0 \
    tensorboardx==2.1

git clone https://github.com/ardaduz/deep-video-mvs.git
pip install -e deep-video-mvs
```

<br />

---
### Data Structure
---
The scripts for parsing the datasets are provided in the **`dataset`** folder.
All of the scripts might not work straight ahead due to naming and foldering conventions while
downloading the datasets, however they should help reduce the effort required. Exporting [ScanNet](http://www.scan-net.org/) .sens 
files, both for training and testing, should work with very minimal effort. The script that is provided here is
a modified version of the official code and similarly requires python2.

During testing, the system expects a data structure for a particular scene as provided in the 
**`sample-data/hololens-dataset/000`**. We assume PNG format for all images.
* **`images`** folder contains the input images that will be used by the model 
and the naming convention is not important. The system considers the sequential order
alphabetically.
* **`depth`** folder contains the groundtruth depth maps that are used for metric evaluation,
the names must match with the color images. The depth images must be uint16 PNG format, 
and the depth value in millimeters. 
For example, if the depth is 1.12 meters for a pixel location,
it should read 1120 in the groundtruth depth image.
* **`poses.txt`** contains CAMERA-TO-WORLD pose corresponding to each color and depth image. 
Each line is one flattened pose in homogeneous coordinates.
* **`K.txt`** is the intrinsic matrix for a given sequence after the images are undistorted.

During training, the system expects each scene to be placed in a folder, and color image and depth image for a time step 
to be packed inside a zipped numpy archive (.npz). See the code [here](https://github.com/ardaduz/deep-video-mvs/blob/master/dataset/scannet-export/scannet-export.py#L151).
We use frame_skip=4 while exporting the ScanNet training and validation scenes due to the large amount of data.
The training/validation split of unique scenes which are used during this work is also provided 
[here](https://github.com/ardaduz/deep-video-mvs/blob/master/dataset/scannet-export),
one may replace the randomly generated ones with these two.

<br />

---
### Training and Testing:
---
* **The pre-trained weights are provided. They are placed 
[here](https://github.com/ardaduz/deep-video-mvs/tree/master/dvmvs/fusionnet/weights) and automatically
loaded during testing.**

* **There are no command line arguments for the system.
Instead, many general parameters are controlled from the `config.py` within the class `Config`.**

* **Please adjust the input and output folder locations (and/or other settings) inside the `config.py`.**

### Training:
In addition to the general **`Config`**, very specific training hyperparameters 
like subsequence length, learning rate, etc. are controlled directly inside 
the training scripts from **`TrainingHyperparameters`** 
[class](https://github.com/ardaduz/deep-video-mvs/blob/master/dvmvs/fusionnet/run-training.py#L18).

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

### Testing:
We provide two scripts for running the inference:

#### 1. Bulk Testing 
First is **`run-testing.py`** for evaluating on multiple datasets and/or sequences 
at a run. This script requires pre-selected keyframe filenames for the desired sequences,
similar to the ones provided in the **`sample-data/indices`**. In 
[a keyframe file](https://github.com/ardaduz/deep-video-mvs/blob/master/sample-data/indices/keyframe%2Bhololens-dataset%2B000%2Bnmeas%2B3), 
each row represents a timestep, the entry in the first column represents the reference frame, and the entries in
the second, third, ... columns represent the measurement frames used for the cost volume computation.
One can determine the keyframe filenames with custom keyframe selection approaches,
or we provide the simulation of our keyframe selection heuristic in
**`simulate_keyframe_buffer.py`**. The predictions and errors of bulk testing 
are saved to the **`Config.test_result_folder`**.

#### 2. Single Scene Online Testing
Second is **`run-testing-online.py`** to run the testing in an online fashion. 
One can specify a single scene in **`Config.test_online_scene_path`**, then run the online inference 
to evaluate on the specified scene.
In this script, we use our keyframe selection heuristic on-the-go and predict the depth maps 
for the selected keyframes (Attention! We do not predict depth maps for all images). 
The predictions and errors of single scene online testing are saved to the working directory.
To run the online testing:
```
cd deep-video-mvs/dvmvs/fusionnet
python run-testing-online.py
``` 

Predicted depth maps for a scene and
average error of each frame are saved in .npz format. 
Errors contain 8 different metrics for each frame in order:
`abs_error`, `abs_relative_error`, `abs_inverse_error`, `squared_relative_error`, 
`rmse`, `ratio_125`, `ratio_125_2`, `ratio_125_3`. They can be accessed with:
```
predictions = numpy.load(prediction_filename)['arr_0']
errors = numpy.load(error_filename)['arr_0']
```

<br />

---
### Comparison with the Existing Methods:
---
In this work, our method is compared with [DELTAS](https://github.com/magicleap/DELTAS), 
[GP-MVS](https://github.com/AaltoML/GP-MVS), [DPSNet](https://github.com/sunghoonim/DPSNet), 
[MVDepthNet](https://github.com/HKUST-Aerial-Robotics/MVDepthNet) and [Neural RGBD](https://github.com/NVlabs/neuralrgbd).
For ease of evaluation, we slightly modified the inference codes of the first four methods to make them 
compatible with the data structure and the keyframe selection files. For Neural RGBD, in contrast, we adjusted 
the data structure and used the original code. The modified inference codes (and the finetuned weights, if necessary)
are provided in the **`dvmvs/baselines`** directory. Please refer to the paper for the comparison results.
