# YotoR: You Transform One Representation
Repository for thesis YotoR, an unified network for object detection using transformers.


Architecture based on [You Only Learn One Representation: Unified Network for Multiple Tasks](https://arxiv.org/abs/2105.04206) and [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/pdf/2103.14030.pdf)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/you-only-learn-one-representation-unified/real-time-object-detection-on-coco)](https://paperswithcode.com/sota/real-time-object-detection-on-coco?p=you-only-learn-one-representation-unified)

## Installation

Docker environment (recommended)
<details><summary> <b>Expand</b> </summary>

```
# create the docker container, you can change the share memory size if you have more.
nvidia-docker run --name swyolor -it -v your_coco_path/:/coco/ -v your_code_path/:/swyolor --shm-size=64g nvcr.io/nvidia/pytorch:20.11-py3

# apt install required packages
apt update
apt install -y zip htop screen libgl1-mesa-glx

# pip install required packages
pip install seaborn thop

# install mish-cuda if you want to use mish activation
# https://github.com/thomasbrandon/mish-cuda
# https://github.com/JunnYu/mish-cuda
cd /
git clone https://github.com/JunnYu/mish-cuda
cd mish-cuda
python setup.py build install

# install pytorch_wavelets if you want to use dwt down-sampling module
# https://github.com/fbcotter/pytorch_wavelets
cd /
git clone https://github.com/fbcotter/pytorch_wavelets
cd pytorch_wavelets
pip install .

# go to code folder
cd /Swin-Yolor
```

</details>

Colab environment
<details><summary> <b>Expand</b> </summary>
  
```
git clone https://github.com/WongKinYiu/yolor
cd yotor

# pip install required packages
pip install -qr requirements.txt

# install mish-cuda if you want to use mish activation
# https://github.com/thomasbrandon/mish-cuda
# https://github.com/JunnYu/mish-cuda
git clone https://github.com/JunnYu/mish-cuda
cd mish-cuda
python setup.py build install
cd ..

# install pytorch_wavelets if you want to use dwt down-sampling module
# https://github.com/fbcotter/pytorch_wavelets
git clone https://github.com/fbcotter/pytorch_wavelets
cd pytorch_wavelets
pip install .
cd ..
```

</details>

Prepare COCO dataset
<details><summary> <b>Expand</b> </summary>

```
cd /yolor
bash scripts/get_coco.sh
```

</details>

Prepare pretrained weight
<details><summary> <b>Expand</b> </summary>

```
cd /yotor
bash scripts/get_pretrain.sh
```

</details>
