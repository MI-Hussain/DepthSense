# DepthSense
This is the official updated version of DepthSense - 
the official implementation of "Advancing Autonomous Driving: DepthSense with Radar and Spatial Attention" paper under revision.




## :wrench: Dependencies and Installation

- Python >= 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
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
@Article{hussain2021rvmde,
    title={RVMDE : Radar Validated Monocular Depth Estimation for Robotics},
    author={Muhammad Ishfaq Hussain, Muhammad Aasim Rafique and Moongu Jeon},
    journal={arXiv:2109.05265v1},
    year={2021}
}
```
