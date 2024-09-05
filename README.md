
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

- **KITTI, nuScenes and Waymo Data**

Follow instructions of [data_setup_README.md](data/data_setup_README.md) to setup KITTI, nuScenes and Waymo as follows:

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

## Training & inference

Train the model:

```bash
chmod +x scripts_training.sh
./scripts_training.sh
```
inference the model:

```bash
chmod +x scripts_inference.sh
./scripts_inference.sh
```
