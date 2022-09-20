# IRNet
The implementation of paper ["Shape Reconstruction of Object-Level Building From Single Image Based on Implicit Representation Network"](https://ieeexplore.ieee.org/document/9606874).


## Installation

You can create an anaconda environment called `IRNet` using
```
conda env create -f environment.yaml
conda activate IRNet
```

Next, compile the extension modules in `src/utils/`.  
You can follow the `readme` documentation in the module folders to compile.

## Datasets

The data folder structure used by this network is the same as [Occupancy Network](https://github.com/autonomousvision/occupancy_networks). If you want to train or test your own data, you can employ its code for processing data and building folder structure, or change the `src/data/` and `dataloader.py` according to your own data.  

The SDF data is generated from the 3D model and the code is employ from the [DISN](https://github.com/laughtervv/DISN).  

The data folder structure is as
```plain
data
├── class
│     ├── building-1
│     |       ├── img_choy2016
│     |       |        ├── 000.jpg           # image 1
│     |       |        ├── 001.jpg           # image 2
│     |       |        ├── 002.jpg           # image 3
│     |       |        ├── 003.jpg           # image 4
│     |       ├── ori_sample.h5              # sdf data
|     ├── building-2
|     ├── building-n
|     ├── ...
|     ├── train.lst                          # building name in train set
|     ├── test.lst                           # building name in test set
|     ├── val.lst                            # building name in val set
```



## Training and Generation

The paths and parameters needed for network training and generation can be modified in the `train.yaml` and `generate.yaml`.

For training the IRNet, you can use
```
python train.py train.yaml
```

For generate predicted building models, you can use
```
python generate.py generate.yaml
```

You can monitor on http://localhost:6006 the training process using tensorboard:
```
cd OUTPUT_DIR
tensorboard --logdir ./logs --port 6006
```


# Futher Information
If you find our code or paper useful, please consider citing

    @ARTICLE{IRNet,  
        author={Zhao, Chunhui and Zhang, Chi and Yan, Yiming and Su, Nan},  
        journal={IEEE Geoscience and Remote Sensing Letters},   
        title={Shape Reconstruction of Object-Level Building From Single Image Based on Implicit Representation Network},   
        year={2022},  
        volume={19},  
        pages={1-5},  
        doi={10.1109/LGRS.2021.3126767}
    }
