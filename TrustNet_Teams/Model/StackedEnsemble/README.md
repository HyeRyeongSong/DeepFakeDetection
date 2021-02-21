lding docker image
All libraries and enviroment is already configured with Dockerfile. It requires docker engine https://docs.docker.com/engine/install/ubuntu/ and  nvidia docker in your system https://github.com/NVIDIA/nvidia-docker.

To build a docker image run `docker build -t df .`

## Running docker 
`docker run --runtime=nvidia --ipc=host -d -it --name dfdc --volume <DATA_ROOT>:/workspace df`

### Data
Download the [deepfake-detection-challenge-data](https://www.kaggle.com/c/deepfake-detection-challenge/data) and extract all files to `TrustNet/Model/StackedEnsemble/data`. This directory must have the following structure:
```
TrustNet/Model/StackedEnsemble/data
├── dfdc_train_part_0
├── dfdc_train_part_1
├── dfdc_train_part_10
├── dfdc_train_part_11
├── dfdc_train_part_12
├── dfdc_train_part_13
├── dfdc_train_part_14
├── dfdc_train_part_15
├── dfdc_train_part_16
├── dfdc_train_part_17
├── dfdc_train_part_18
├── dfdc_train_part_19
├── dfdc_train_part_2
├── dfdc_train_part_20
├── dfdc_train_part_21
├── dfdc_train_part_22
├── dfdc_train_part_23
├── dfdc_train_part_24
├── dfdc_train_part_25
├── dfdc_train_part_26
├── dfdc_train_part_27
├── dfdc_train_part_28
├── dfdc_train_part_29
├── dfdc_train_part_3
├── dfdc_train_part_30
├── dfdc_train_part_31
├── dfdc_train_part_32
├── dfdc_train_part_33
├── dfdc_train_part_34
├── dfdc_train_part_35
├── dfdc_train_part_36
├── dfdc_train_part_37
├── dfdc_train_part_38
├── dfdc_train_part_39
├── dfdc_train_part_4
├── dfdc_train_part_40
├── dfdc_train_part_41
├── dfdc_train_part_42
├── dfdc_train_part_43
├── dfdc_train_part_44
├── dfdc_train_part_45
├── dfdc_train_part_46
├── dfdc_train_part_47
├── dfdc_train_part_48
├── dfdc_train_part_49
├── dfdc_train_part_5
├── dfdc_train_part_6
├── dfdc_train_part_7
├── dfdc_train_part_8
├── dfdc_train_part_9
└── test_videos
```

## The hardware we used
- CPU: Intel(R) Xeon(R) Gold 5120 CPU @ 2.20GHz
- RAM: 180 GB
- GPU: NVIDIA Tesla V100 SXM2 32 GB x 2
- SSD: 2 TB
