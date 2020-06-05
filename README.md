![Python >=3.5](https://img.shields.io/badge/Python->=3.5-blue.svg)
![PyTorch >=1.0](https://img.shields.io/badge/PyTorch->=1.0-yellow.svg)

# Self-paced Contrastive Learning

[[Paper]](https://arxiv.org/abs/2006.02713) [[Project]](https://yxgeee.github.io/projects/spcl.html)

This repository contains the implementation of [Self-paced Contrastive Learning with Hybrid Memory for Domain Adaptive Object Re-ID](https://arxiv.org/abs/2006.02713), which provides state-of-the-art performances on both **unsupervised domain adaptation** tasks and **unsupervised learning** tasks for object re-ID, including person re-ID and vehicle re-ID.

![framework](figs/framework.png)


## Requirements

### Installation

```shell
git clone https://github.com/yxgeee/SpCL.git
cd SpCL
python setup.py install
```

### Prepare Datasets

```shell
cd examples && mkdir data
```
Download the person datasets [DukeMTMC-reID](https://arxiv.org/abs/1609.01775), [Market-1501](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view), [MSMT17](https://arxiv.org/abs/1711.08565), [PersonX](https://github.com/sxzrt/Instructions-of-the-PersonX-dataset#a-more-chanllenging-subset-of-personx), and the vehicle datasets [VehicleID](https://www.pkuml.org/resources/pku-vehicleid.html), [VeRi-776](https://github.com/JDAI-CV/VeRidataset), [VehicleX](https://www.aicitychallenge.org/2020-track2-download/).
Then unzip them under the directory like
```
SpCL/examples/data
├── dukemtmc
│   └── DukeMTMC-reID
├── market1501
│   └── Market-1501-v15.09.15
├── msmt17
│   └── MSMT17_V1
├── personx
│   └── PersonX
├── vehicleid
│   └── VehicleID -> VehicleID_V1.0
├── vehiclex
│   └── AIC20_ReID_Simulation -> AIC20_track2/AIC20_ReID_Simulation
└── veri
    └── VeRi -> VeRi_with_plate
```

### Prepare Pre-trained Models for IBN-Net
When training with the backbone of [IBN-ResNet](https://arxiv.org/abs/1807.09441), you need to download the ImageNet-pretrained model from this [link](https://drive.google.com/drive/folders/1thS2B8UOSBi_cJX6zRy6YYRwz_nVFI_S) and save it under the path of `logs/pretrained/`.
```shell
mkdir logs && cd logs
mkdir pretrained
```
The file tree should be
```
SpCL/logs
└── pretrained
    └── resnet50_ibn_a.pth.tar
```
ImageNet-pretrained models for **ResNet-50** will be automatically downloaded in the python script.


## Training

We utilize 4 GTX-1080TI GPUs for training. **Note that**

+ use `--iters 400` (default) for DukeMTMC-reID, Market-1501 and PersonX datasets, and `--iters 800` for MSMT17, VeRi-776, VehicleID and VehicleX datasets;
+ use `--width 128 --height 256` (default) for person datasets, and `--height 224 --width 224` for vehicle datasets;
+ use `-a resnet50` (default) for the backbone of ResNet-50, and `-a resnet_ibn50a` for the backbone of IBN-ResNet.

### Unsupervised Domain Adaptation
To train the model(s) in the paper, run this command:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/spcl_train_uda.py -ds $SOURCE_DATASET -dt $TARGET_DATASET --logs-dir $PATH_LOGS
```

*Example #1:* DukeMTMC-reID -> Market-1501
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/spcl_train_uda.py -ds dukemtmc -dt market1501 --logs-dir logs/spcl_uda/duke2market_resnet50
```
*Example #2:* DukeMTMC-reID -> MSMT17
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/spcl_train_uda.py -ds dukemtmc -dt msmt17 --iters 800 --logs-dir logs/spcl_uda/duke2msmt_resnet50
```
*Example #3:* VehicleID -> VeRi
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/spcl_train_uda.py -ds vehicleid -dt veri --iters 800 --height 224 --width 224 --logs-dir logs/spcl_uda/vehicleid2veri_resnet50
```

### Unsupervised Learning
To train the model(s) in the paper, run this command:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/spcl_train_usl.py -d $DATASET --logs-dir $PATH_LOGS
```

*Example #1:* DukeMTMC-reID
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/spcl_train_usl.py -d dukemtmc --logs-dir logs/spcl_usl/duke_resnet50
```


## Evaluation

We utilize 1 GTX-1080TI GPU for testing. **Note that**

+ use `--width 128 --height 256` (default) for person datasets, and `--height 224 --width 224` for vehicle datasets;
+ use `--dsbn` for domain adaptive models, and add `--test-source` if you want to test on the source domain;
+ use `-a resnet50` (default) for the backbone of ResNet-50, and `-a resnet_ibn50a` for the backbone of IBN-ResNet.

### Unsupervised Domain Adaptation

To evaluate the model on the target-domain dataset, run:

```shell
CUDA_VISIBLE_DEVICES=0 python examples/test.py --dsbn -d $DATASET --resume $PATH_MODEL
```

To evaluate the model on the source-domain dataset, run:

```shell
CUDA_VISIBLE_DEVICES=0 python examples/test.py --dsbn --test-source -d $DATASET --resume $PATH_MODEL
```

*Example #1:* DukeMTMC-reID -> Market-1501
```shell
# test on the target domain
CUDA_VISIBLE_DEVICES=0 python examples/test.py --dsbn -d market1501 --resume logs/spcl_uda/duke2market_resnet50/model_best.pth.tar
# test on the source domain
CUDA_VISIBLE_DEVICES=0 python examples/test.py --dsbn --test-source -d dukemtmc --resume logs/spcl_uda/duke2market_resnet50/model_best.pth.tar
```

### Unsupervised Learning
To evaluate the model, run:
```shell
CUDA_VISIBLE_DEVICES=0 python examples/test.py -d $DATASET --resume $PATH
```

*Example #1:* DukeMTMC-reID
```shell
CUDA_VISIBLE_DEVICES=0 python examples/test.py -d dukemtmc --resume logs/spcl_usl/duke_resnet50/model_best.pth.tar
```

## Trained Models

![framework](figs/results.png)

You can download the above models in the paper from [Google Drive](https://drive.google.com/open?id=19vYA4EfInuH4ZKg0HeBRmDmgK1KLdivz).


## Citation
If you find this code useful for your research, please cite our paper
```
@misc{ge2020selfpaced,
    title={Self-paced Contrastive Learning with Hybrid Memory for Domain Adaptive Object Re-ID},
    author={Yixiao Ge and Dapeng Chen and Feng Zhu and Rui Zhao and Hongsheng Li},
    year={2020},
    eprint={2006.02713},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
