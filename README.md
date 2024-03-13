# YotoR: You Only Transform One Representation
Repository for YotoR, an unified network for object detection using transformers.


Architecture based on [You Only Learn One Representation: Unified Network for Multiple Tasks](https://arxiv.org/abs/2105.04206) and [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/pdf/2103.14030.pdf)

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
pip install seaborn thop timm

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
cd /YotoR
```

</details>

Colab environment
<details><summary> <b>Expand</b> </summary>
  
```
apt update
apt install -y zip htop screen libgl1-mesa-glx

cd YotoR

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
cd /YotoR
bash scripts/get_coco.sh
```

</details>

Prepare pretrained weights
<details><summary> <b>Expand</b> </summary>

```
pip install gdown
gdown --id 1kgxZcOo1PUBo1Q6wFrbsAg0JsiJnz_ra
gdown --id 184Ed_y-QvB6ulC3-Y5Qhm8M552Vj4K27
gdown --id 1fTwLWHLuwPZTfJXaCh1WY4uEZ8smg8-6
gdown --id 1dkW_l9yK6tsaHQIR9aCa2S1_e8KYBDxm
```

</details>

Evaluate models
<details><summary> <b>Expand</b> </summary>

```
cd /YotoR
python testing.py --data data/coco.yaml --task 'test' --img 1280 --batch 1 --conf 0.001 --iou 0.65 --device 0 --cfg cfg/yolor_p6.cfg --weights 'best_yolor_p6.pt' --name yolor_p6_test --save-txt --save-json
python testing.py --data data/coco.yaml --task 'test' --img 1280 --batch 1 --conf 0.001 --iou 0.65 --device 0 --cfg cfg/yotor_Tp5.cfg --weights 'best_tp5.pt' --name yotor_tp5_test```
python testing.py --data data/coco.yaml --task 'test' --img 1280 --batch 1 --conf 0.001 --iou 0.65 --device 0 --cfg cfg/yotor_Bp4.cfg --weights 'best_bp4.pt' --name yotor_bp4_test
python testing.py --data data/coco.yaml --task 'test' --img 1280 --batch 1 --conf 0.001 --iou 0.65 --device 0 --cfg cfg/yotor_Bb4.cfg --weights 'best_bb4.pt' --name yotor_bb4_test
</details>
