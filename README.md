<h1 align="center">
SDUI4.0 Bin Picking
</h1>

The code used for obtaining grasp poses for the SDUI4.0 Bin Picking project. The code is used in the following papers:

<a href="https://arxiv.org/abs/2409.11512">[Good Grasps Only: A data engine for self-supervised fine-tuning of pose estimation using grasp poses for verification]</a>

<a href="https://arxiv.org/abs/2309.16221">[Off-the-shelf bin picking workcell with visual pose estimation: A case study on the world robot summit 2018 kitting task]</a>

# To test the code:

>		python -m sdui40binpick.compute_grasp_poses_in_bin config_wrs.json data/wrs4/depth_image_01mm_resolution.png data/wrs4/color.png

### Config File

In config.py camera parameters, bin cad model, bin transform and object information.

The bin transform can be found using:

>		python sdui40binpick/select_bin_position.py config_wrs.json data/NormalRackBin.stl data/wrs4/depth_image_01mm_resolution.png data/wrs4/color.png

# Installation

Python 3.9.15

First download and install <a href="https://github.com/fhagelskjaer/keymatchnet">[KeyMatchNet]</a>

Then install the following packages:

>		pip install opencv-python
>		pip install python-fcl
>		pip install pyrender==0.1.45

# Running the zivid camera

In the folder "utils" there is a script to capture depth and color data from the zivid camera.

> python sdui40binpick/capture_pc_and_depth_in_folder.py

Before running the script install the zivid drivers from and then install the python package:

>		pip install zivid

# Citation
If you use this code in your research, please cite the paper:

```
@INPROCEEDINGS{10597534,
  author={Hagelskjær, Frederik and Lorenzen, Kasper H⊘j and Kraft, Dirk},
  booktitle={2024 21st International Conference on Ubiquitous Robots (UR)}, 
  title={Off-the-Shelf Bin Picking Workcell with Visual Pose Estimation: A Case Study on the World Robot Summit 2018 Kitting Task}, 
  year={2024},
  volume={},
  number={},
  pages={145-152},
  keywords={Point cloud compression;Visualization;Pose estimation;Training data;Vision sensors;Robot sensing systems;Task analysis},
  doi={10.1109/UR61395.2024.10597534}}
```
