# Deepfake Detection Challenge
## Prerequisites
### Environment
Use the docker to get an environment close to what was used in the training. Run the following command to build the docker image:
```bash
cd [solution folder]
sudo docker build -t dfdc .
```
### Data
Download the [deepfake-detection-challenge-data](https://www.kaggle.com/c/deepfake-detection-challenge/data) and extract all files to `solution/dsfacedetector/data`. This directory must have the following structure:
```
solution/dsfacedetector/data
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
├── dfdc_train_part_4
├── dfdc_train_part_5
├── dfdc_train_part_6
├── dfdc_train_part_7
├── dfdc_train_part_8
├── dfdc_train_part_9
└── test_videos
        ├── dfdc_train_part_36
        ├── dfdc_train_part_37
        ├── dfdc_train_part_38
        ├── dfdc_train_part_39
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
```

### External data
According to the rules of the competition, external data is allowed. The solution does not use other external data, except for pre-trained models. Below is a table with information about these models.

| File Name | Source | Direct Link | Forum Post |
| --------- | ------ | ----------- | ---------- |
| WIDERFace_DSFD_RES152.pth | [github](https://github.com/Tencent/FaceDetection-DSFD/tree/31aa8bdeaf01a0c408adaf2709754a16b17aec79) | [google drive](https://drive.google.com/file/d/1WeXlNYsM6dMP3xQQELI-4gxhwKUQxc3-/view) | [link](https://www.kaggle.com/c/deepfake-detection-challenge/discussion/121203#761391) |
| noisy_student_efficientnet-b7.tar.gz | [github](https://github.com/tensorflow/tpu/tree/4719695c9128622fb26dedb19ea19bd9d1ee3177/models/official/efficientnet) | [link](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/noisystudent/noisy_student_efficientnet-b7.tar.gz) | [link](https://www.kaggle.com/c/deepfake-detection-challenge/discussion/121203#748358) |  

Download these files and copy them to the `external_data` folder. 

## How to train the model
Run the docker container with the paths correctly mounted:
```bash
sudo docker run --runtime=nvidia -i -t -d --rm --ipc=host -v solution/dsfacedetector/data:/kaggle/input/deepfake-detection-challenge:ro -v solution:/kaggle/solution --name dfdc dfdc
sudo docker exec -it dfdc /bin/bash
cd /kaggle/solution
```
Convert pre-trained model from tensorflow to pytorch:
```bash
bash convert_tf_to_pt.sh
```
Detect faces on videos:
```bash
python3.6 detect_faces_on_videos.py
```
_Note: You can parallelize this operation using the `--part` and `--num_parts` arguments_  
Generate tracks:
```bash
python3.6 generate_tracks.py
```
Generate aligned tracks:
```bash
python3.6 generate_aligned_tracks.py
```
Extract tracks from videos:
```bash
python3.6 extract_tracks_from_videos.py
```
_Note: You can parallelize this operation using the `--part` and `--num_parts` arguments_  
Generate track pairs:
```bash
python3.6 generate_track_pairs.py
```
Train models:
```bash
python3.6 train_b7_ns_seq_aa_original_100k.py
python3.6 train_b7_ns_seq_aa_original_100k_380v2.py
python3.6 train_b7_ns_seq_aa_original_100k_416.py
```
Copy the final weights and convert them to FP16:
```bash
python3.6 copy_weights_seq.py
python3.6 copy_weights_seq380.py
python3.6 copy_weights_seq416.py
```
## How to generate submission
Run the following command
```bash
python3.6 predict_seq.py
python3.6 predict_seq380.py
python3.6 predict_seq416.py
```
## The hardware we used
- CPU: Intel(R) Xeon(R) Gold 5120 CPU @ 2.20GHz
- RAM: 180 GB
- GPU: NVIDIA Tesla V100 SXM2 32 GB x 2
- SSD: 2 TB

