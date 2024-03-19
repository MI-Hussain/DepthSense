# RVMDE-II
This is the official updated version of RVMDE - 
the official implementation of "Radar Validated Monocular Depth Completion for Robotics and Autonomous Driving," available at https://arxiv.org/abs/2109.05265v1.

# Abstract
Stereoscopy naturally captures depth perception in a scene, creating an involuntary phenomenon in our perception of the 3D world. However, achieving precise depth estimation requires meticulous calibration of binocular vision sensors. In contrast, a monocular camera offers a solution, albeit with reduced accuracy in depth estimation, particularly under challenging imaging conditions. Additionally, optical sensors encounter difficulties in collecting data in harsh environments, prompting the exploration of radar as an alternative source, providing coarse yet highly accurate signals. This study investigates the effectiveness of integrating radar-derived coarse signals with fine-grained data from a monocular camera for depth estimation across various environmental conditions. Furthermore, the study proposes a variant of the Feature Pyramid Network (VFPN) that leverages fine-grained image features at multiple scales while reducing the number of parameters. The VFPN combines image feature maps with radar data features extracted through a convolutional neural network. These concatenated hierarchical features are then used for depth prediction via ordinal regression. Experiments conducted on the nuScenes dataset demonstrate that our proposed architecture consistently outperforms competitors in quantitative evaluations. This is achieved while maintaining reduced parameter complexity and ensuring faster inference times. The results of our depth estimation suggest that these techniques could serve as a viable alternative to traditional stereo depth estimation in critical applications within the realms of robotics and self-driving vehicles. The code for this study can be found at: https://github.com/MI-Hussain/RVMDE-II.


## :wrench: Dependencies and Installation

- Python >= 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.9](https://pytorch.org/)

### Installation

1. Clone repo

    ```bash
    git clone https://github.com/MI-Hussain/RVMDE-II
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
rvmde/
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

Please follow external repos (https://github.com/lochenchou/DORN_radar) for Height Extension and (https://github.com/longyunf/rc-pda) for RVMDE with MER's to generte the dataset for training and evaluation.

### Evaluation for RVMDE-II on nuScenes

Download [pre-trained weights](https://drive.google.com/file/d/1VKVg63d5UMNjc2busvdM23rXrs8TZb-X/view?usp=sharing)


Modifying dataset path in `valid_loader.py`, evalutation list path in `data_loader.py`, pretrained_weights path in Evalutation_rvmde.py file to evalute. 

For evaluation on interms of day,night,rain change the list path first. The evaluation lists are saved in .\list directory.

``` bash
jupyter notebook
Evaluate_rvmde.ipynb     #Test
```

# RVMDE-II with MER's Evalution

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
