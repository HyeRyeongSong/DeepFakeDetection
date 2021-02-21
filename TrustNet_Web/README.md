# TrustTube Server

[![Version](https://img.shields.io/badge/TrustAPI-v0.0.1-brightgreen)](https://git.swmgit.org/swmaestro/trustnet-2)
[![PyTorch](https://img.shields.io/badge/PyTorch-v1.7.0-brightgreen)](https://github.com/pytorch/pytorch/releases/tag/v1.7.0)
[![TensorRT](https://img.shields.io/badge/TensorRT-v7-brightgreen)](https://github.com/NVIDIA/TensorRT/releases/tag/20.10)
[![OpenCV](https://img.shields.io/badge/OpenCV-v4.4.0-brightgreen)](https://github.com/opencv/opencv/releases/tag/4.4.0)
[![Flask](https://img.shields.io/badge/Flask-v1.1.2-brightgreen)](https://github.com/pallets/flask/releases/tag/1.1.2)
[![TritonClient](https://img.shields.io/badge/-Triton_Client-blue)](https://github.com/triton-inference-server/server)

## How to run

### Building docker image & Running docker

```bash
docker build -t trustnetapi .
docker run --name trustnetapi --gpus all -it -d --net=host --ipc=host -v <your workspace>:/to/path/dir/ trustnetapi
```

### Instructions Python packages

```bash
pip3 install -r requirements.txt
```

### Run

```bash
./run
```




## TrustTube Preview

### TrustTube Main Page

![Main Homepage](images/Homepage.png)

### TrustTube Register Page

![Register Page](images/Homepage5.png)

### TrustTube Login Page

![Login Page](images/Homepage4.png)

### TrustTube Upload Page

![Upload Page](images/Homepage3.png)


### DeepFake Detected page with (Class Activation Map)

![Result Page](images/Homepage2.png)



## Directory Structure :

	|-- Dockerfile
	|-- README.md
	|-- api
	|   |-- TrustNetAPI.py
	|   |-- classifiers.py
	|   |-- deepfake_utils.py
	|   |-- grad_cam.py
	|   `-- weights
	|-- app
	|   |-- __init__.py
	|   |-- forms.py
	|   |-- models.py
	|   |-- routes.py
	|   |-- static
	|   |   |-- avatar
	|   |   |   |-- avatar.png
	|   |   |   `-- student.png
	|   |   |-- cover_pics
	|   |   |   `-- cover.png
	|   |   |-- css
	|   |   |   |-- kopubdotum.css
	|   |   |   |-- main.css
	|   |   |   `-- style.css
	|   |   |-- logo
	|   |   |   |-- sw_maestro.png
	|   |   |   `-- trustNet.png
	|   |   `-- videos
	|   `-- templates
	|       |-- CAM.html
	|       |-- account.html
	|       |-- home.html
	|       |-- layout.html
	|       |-- login.html
	|       |-- register.html
	|       |-- update_video.html
	|       |-- upload.html
	|       `-- video.html
	|-- config.py
	|-- images
	|   |-- Homepage.png
	|   |-- Homepage1.png
	|   |-- Homepage2.png
	|   |-- Homepage3.png
	|   |-- Homepage4.png
	|   `-- Homepage5.png
	|-- migrations
	|   |-- alembic.ini
	|   |-- env.py
	|   |-- script.py.mako
	|   `-- versions
	|       `-- b0e1bb55d78d_create_models.py
	|-- requirements.txt
	|-- reset
	|-- run
	|-- server.py
	|-- youtube.code-workspace
	`-- youtube.db
