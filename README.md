
## Setup

- **Requirements**

    1. Python 3.7
    2. [PyTorch](http://pytorch.org) 1.10
    3. Torchvision 0.11
    4. Cuda 11.3
    5. Ubuntu 18.04/Debian 8.9


- **Python**

```
pip install opencv-python pandas
```

- **KITTI, and Waymo Data**

Follow instructions of [data_setup_README.md](data/data_setup_README.md) to setup KITTI, and Waymo as follows:

```bash
DEVIANT
├── data
│      ├── KITTI
│      │      ├── ImageSets
│      │      ├── kitti_split1
│      │      ├── training
│      │      │     ├── calib
│      │      │     ├── image_2
│      │      │     └── label_2
│      │      │
│      │      └── testing
│      │            ├── calib
│      │            └── image_2
│      └── waymo
│             ├── ImageSets
│             ├── training
│             │     ├── calib
│             │     ├── image
│             │     └── label
│             │
│             └── validation
│                   ├── calib
│                   ├── image
│                   └── label
├── output
│      └── pretrained_model.pth
│
├── experiments
├── images
├── lib   
│ ...
```



```bash
sudo apt-get install libopenblas-dev libboost-dev libboost-all-dev gfortran
sh data/KITTI/kitti_split1/devkit/cpp/build.sh
```

```
pip install tensorflow-gpu==2.4
pip install pandas
pip3 install waymo-open-dataset-tf-2-4-0 --user
```
### Model Zoo

We provide logs/models/predictions for the main experiments on KITTI Val /KITTI Test/Waymo Val data splits available to download here.

| Data_Splits | Method  | Config<br/>(Run)                                          | Weight<br>/Pred  |
|------------|-------------|------------------------------------------------------------------|----------|
| KITTI Val  |   GUP Net   | [run_201](experiments/config_run_201_a100_v0_1.yaml) | [gdrive](https://drive.google.com/file/d/17qezmIjckRSAva1fNnYBmgR9LaY-dPnp/view?usp=sharing) 
| KITTI Val  |   DEVIANT   | [run_221](experiments/run_221.yaml)                  | [gdrive](https://drive.google.com/file/d/1CBJf8keOutXVSAiu9Fj7XQPQftNYC1qv/view?usp=sharing)
| KITTI Val  |   GUP Net   | [run_201](experiments/config_run_201_a100_v0_1.yaml) | [gdrive](https://drive.google.com/file/d/17qezmIjckRSAva1fNnYBmgR9LaY-dPnp/view?usp=sharing) 
| Waymo Val  |   GUP Net   | [run_1050](experiments/run_1050.yaml)                | [gdrive](https://drive.google.com/file/d/1wuTTuZrFVsEv4ttQ0r3X_s8D3OjYE84E/view?usp=sharing)     
| Waymo Val  |   DEVIANT   | [run_1051](experiments/run_1051.yaml)                | [gdrive](https://drive.google.com/file/d/1ixCVS85yVU9k6kuHrcYw_qJoy9Z4d0FD/view?usp=sharing)     
| Waymo Val  | GUP Net(KD) | [run_1050](experiments/run_1050.yaml)                | [gdrive](https://drive.google.com/file/d/1VJQ7tKW_HdpfR7adqWS6PzM8gnIn9zmP/view?usp=drive_link)
| Waymo Val  | DEVIANT(KD) | [run_1051](experiments/run_1051.yaml)                | [gdrive](https://drive.google.com/file/d/1u6EZpOypW217YXYQzgo-88V1XF_zonQZ/view?usp=drive_link)     
## Training & inference

Train the model:

```bash
python -u tools/train_val.py --config=experiments/pretrained.pth
```
inference the model:

```bash
python -u tools/train_val.py --config=experiments/pretrained.pth --resume_model output/'config'/checkpoints/checkpoint_epoch_30.pth -e
```
