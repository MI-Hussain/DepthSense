# DepthSense - This is the official updated version of DepthSense   
# The official implementation of "Advancing Autonomous Driving: DepthSense with Radar and Spatial Attention".

Depth perception is crucial for spatial understanding and has traditionally been achieved through stereoscopic imaging. However, the precision of depth estimation using stereoscopic methods depends on the accurate calibration of binocular vision sensors. Monocular cameras, while more accessible, often suffer from reduced accuracy, especially under challenging imaging conditions. Optical sensors, too, face limitations in adverse environments, leading researchers to explore radar technology as a reliable alternative. Although radar provides coarse but accurate signals, its integration with fine-grained monocular camera data remains underexplored. In this research, we propose DepthSense, a novel radar-assisted monocular depth enhancement approach. DepthSense employs an encoder-decoder architecture, a Radar Residual Network, feature fusion with a spatial attention mechanism, and an ordinal regression layer to deliver precise depth estimations. We conducted extensive experiments on the nuScenes dataset to validate the effectiveness of DepthSense. Our methodology not only surpasses existing approaches in quantitative performance but also reduces parameter complexity and inference times. Our findings demonstrate that DepthSense represents a significant advancement over traditional stereo methods, offering a robust and efficient solution for depth estimation in autonomous driving. By leveraging the complementary strengths of radar and monocular camera data, DepthSense sets a new benchmark in the field, paving the way for more reliable and accurate spatial perception systems.



## :wrench: Dependencies and Installation

- Python >= 3.9 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.9](https://pytorch.org/)

### Installation

1. Clone repo

    ```bash
    git clone https://github.com/MI-Hussain/DepthSense
    ```

1. Install dependent packages

```bash
pypardiso
tensorboardX
nuscenes-devkit
```
## Dataset

Download [nuScenes dataset](https://www.nuscenes.org/) (Full dataset (v1.0)) into data/nuscenes/

## Directories
```plain
DepthSense/
    data/                           							 
        nuscenes/                 		    
                annotations/
                maps/
                samples/
                sweeps/
                v1.0-trainval/
    dataloader/
    list/
    result/ download and copy the pretrained model here
    model/                   				   	        
                   	     				
```

### Dataset Prepration use the externel repos

Please follow external repos (https://github.com/lochenchou/DORN_radar) for Height Extension and (https://github.com/longyunf/rc-pda) for DepthSense with MER's to generte the dataset for training and evaluation.

### Evaluation for DepthSense on nuScenes

Download [pre-trained weights](https://drive.google.com/file/d/1VKVg63d5UMNjc2busvdM23rXrs8TZb-X/view?usp=sharing)


Modifying dataset path in `valid_loader.py`, evalutation list path in `data_loader.py`, pretrained_weights path in Evalutation_rvmde.py file to evalute. 

For evaluation on interms of day,night,rain change the list path first. The evaluation lists are saved in .\list directory.

``` bash
jupyter notebook
Evaluate_rvmde.ipynb     #Test
```

### DepthSense with MER's Evalution

Will be updated soon!!!

## Citation
```plain
@ARTICLE{10752892,
  author={Hussain, Muhammad Ishfaq and Naz, Zubia and Rafique, Muhammad Aasim and Jeon, Moongu},
  journal={IEEE Sensors Journal}, 
  title={Advancing Autonomous Driving: DepthSense with Radar and Spatial Attention}, 
  year={2024},
  keywords={Feature Pyramid Network;Monocular Depth Estimation;Radar Data Augmentation;Sensor Fusion},
  doi={10.1109/JSEN.2024.3493196}}
```
