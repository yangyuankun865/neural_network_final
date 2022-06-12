# Exploration of Cutout, Mixup and Cutmix

## Description

This project is an implementation of comparison between Resnet and Swin Transformer based on CIFAR 100 dataset.

## Getting Started

### Dependencies

* Python 3.8

### Installing

* Download this program through git clone and put it in your repository
* Create file plot for all visualization

### Executing program
* You can refer to this environment setting
```
cd your_direction
conda create --name neuralnetwork --clone base
conda activate neuralnetwork 
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install -c conda-forge tensorboardx
pip install torchsummary
pip install einops
pip install timm
```


* Run the training code directly to train Resnet 18 with different random seed
```
python experiment.py
```

* Run the training code directly to train Swin Transformer, --is_SPT will set SPT method and --is_LSA will set LSA method
```
python main.py --model swin 
python main.py --model swin --is_SPT
python main.py --model swin --is_LSA 
python main.py --model swin --is_LSA --is_SPT

```

* Run the code for all tensorboard results
``` 
tensorboard --logdir=tensorboard_direction --bind_all
```

* checkpoint link: https://pan.baidu.com/s/1ysxRKhGiNAC1WBh9Q0b0dw?pwd=0p5y code: 0p5y
