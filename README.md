# MTV:Multi-view-Thermal-Visible-Image-Dataset
This dataset is used for the task of thermal and visible image matching, and thermal image localization. It contains 5 different scenes, such as "building", "park", "countryside", etc.

## Data source

All images and accompanying raw data are collected by ourselves using a DJI H20T camera and PSDK102S mounted on a DJI M300 RTK drone.

## Data content

We provide pose for each thermal image, pose and depth map for the visible image, and the pairing relationship between them with relative path is made into .npz file for easy reading.

## Data format

Datasets are organized in folders according to different scenarios. Each folder contains the following files:

```
MTV/
├── building/
│      ├── images/             # all images
│      ├── depths/             # depth map of visible images
│      ├── spares/             # model file of scene in COLMAP format
│      ├── T_pose/             # Thermal pose files
│      ├── xx_train.npz        # train file
│      ├── xx_val.npz          # test file 
│      └── xx_test.npz         # test file
├── park
│   └── ...
└── villa
    └── ...
```


## Dataset usage

- Download method: Click [Baidu Cloud](https://pan.baidu.com/s/1k_J4N3gZQxzWSkmvmbUxiw) to download. Code:0328.
- License agreement: This dataset follows [MIT License](https://opensource.org/license/mit/).
- Citation format: If you use this dataset in your research or publication, please cite it as follows:
```
@article{liu2022multi,
  title={A Multi-View Thermal--Visible Image Dataset for Cross-Spectral Matching},
  author={Liu, Yuxiang and Liu, Yu and Yan, Shen and Chen, Chen and Zhong, Jikun and Peng, Yang and Zhang, Maojun},
  journal={Remote Sensing},
  volume={15},
  number={1},
  pages={174},
  year={2022},
  publisher={MDPI}
}
```
## Dataset contributors

This dataset is created and maintained by Yuxiang Liu. If you have any questions or suggestions, please contact <liuyuxiang17@nudt.edu.cn>.

## Other information

- Dataset update record:

|Date|Version|Description|
|---|---|---|
|March 21st, 2023|1.0|Release initial version|

