## Building docker image
All libraries and enviroment is already configured with Dockerfile. It requires docker engine https://docs.docker.com/engine/install/ubuntu/ and  nvidia docker in your system https://github.com/NVIDIA/nvidia-docker.

To build a docker image run `docker build -t df .`

## Running docker 
`docker run --runtime=nvidia --ipc=host --rm  --volume <DATA_ROOT>:/dataset -it df`

## Data preparation

Once DFDC dataset is downloaded all the scripts expect to have `dfdc_train_xxx` folders under data root directory. 

Preprocessing is done in a single script **`preprocess_data.sh`** which requires dataset directory as first argument. 
It will execute the steps below:  

##### 1. Find face bboxes
To extract face bboxes I used facenet library, basically only MTCNN. 
`python preprocessing/detect_original_faces.py --root-dir DATA_ROOT`
This script will detect faces in real videos and store them as jsons in DATA_ROOT/bboxes directory

##### 2. Extract crops from videos
To extract image crops I used bboxes saved before. It will use bounding boxes from original videos for face videos as well.
`python preprocessing/extract_crops.py --root-dir DATA_ROOT --crops-dir crops`
This script will extract face crops from videos and save them in DATA_ROOT/crops directory
 
##### 3. Generate landmarks
From the saved crops it is quite fast to process crops with MTCNN and extract landmarks  
`python preprocessing/generate_landmarks.py --root-dir DATA_ROOT`
This script will extract landmarks and save them in DATA_ROOT/landmarks directory
 
##### 4. Generate diff SSIM masks
`python preprocessing/generate_diffs.py --root-dir DATA_ROOT`
This script will extract SSIM difference masks between real and fake images and save them in DATA_ROOT/diffs directory

##### 5. Generate folds
`python preprocessing/generate_folds.py --root-dir DATA_ROOT --out folds.csv`
By default it will use 16 splits to have 0-2 folders as a holdout set. Though only 400 videos can be used for validation as well. 


## Training

Training 5 B7 models with different seeds is done in **`train.sh`** script.

During training checkpoints are saved for every epoch.

## Hardware requirements
Mostly trained on 2xTesla V100 GPUs, thanks to NIPA and AI Hub where I got these gpus https://aihub.or.kr/
Overall training requires 2 GPUs with 12gb+ memory. 
Batch size needs to be adjusted for standard 1080Ti or 2080Ti graphic cards.

As I computed fake loss and real loss separately inside each batch, results might be better with larger batch size, for example on V100 gpus. 
Even though SyncBN is used larger batch on each GPU will lead to less noise as DFDC dataset has some fakes where face detector failed and face crops are not really fakes.   

## Plotting losses to select checkpoints

`python plot_loss.py --log-file logs/<log file>`

![loss plot](images/loss_plot.png "Weighted loss")

## Inference


Kernel is reproduced with `predict_folder.py` script.

## The hardware we used
- CPU: Intel(R) Xeon(R) Gold 5120 CPU @ 2.20GHz
- RAM: 180 GB
- GPU: NVIDIA Tesla V100 SXM2 32 GB x 2
- SSD: 2 TB
