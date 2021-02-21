ARG TENSORRTDOCKER="20.10"

FROM nvcr.io/nvidia/tensorrt:${TENSORRTDOCKER}-py3 

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6 nano mc glances vim \
					 pkg-config libjpeg-dev libtiff5-dev libpng-dev ffmpeg libavcodec-dev libavformat-dev libswscale-dev libxvidcore-dev libx264-dev libxine2-dev \
					 mesa-utils libgl1-mesa-dri libgtkgl2.0-dev libgtkglext1-dev \
					 libatlas-base-dev gfortran libeigen3-dev python3-dev python3-numpy \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Install Pip
RUN python3 -m pip install --upgrade pip
RUN apt-get update -y
RUN apt-get install build-essential cmake -y
RUN apt-get install libopenblas-dev liblapack-dev -y
RUN apt-get install libx11-dev libgtk-3-dev -y
RUN pip install dlib 
RUN pip install facenet-pytorch

WORKDIR /workspace/opencv

RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/4.4.0.zip
RUN unzip opencv.zip
RUN wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.4.0.zip
RUN unzip opencv_contrib.zip

WORKDIR /workspace/opencv/opencv-4.4.0/build

RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_C_COMPILER=/usr/bin/gcc-7 \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D INSTALL_PYTHON_EXAMPLES=ON \
-D INSTALL_C_EXAMPLES=ON \
-D BUILD_DOCS=OFF \
-D BUILD_PERF_TESTS=OFF \
-D BUILD_TESTS=OFF \
-D BUILD_PACKAGE=OFF \
-D BUILD_EXAMPLES=OFF \
-D WITH_TBB=ON \
-D ENABLE_FAST_MATH=1 \
-D CUDA_FAST_MATH=1 \
-D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-11.1 \
-D WITH_CUDA=ON \
-D WITH_CUBLAS=ON \
-D WITH_CUFFT=ON \
-D WITH_NVCUVID=ON \
-D WITH_IPP=OFF \
-D WITH_V4L=ON \
-D WITH_1394=OFF \
-D WITH_GTK=ON \
-D WITH_QT=OFF \
-D WITH_OPENGL=ON \
-D WITH_EIGEN=ON \
-D WITH_FFMPEG=ON \
-D WITH_GSTREAMER=ON \
-D BUILD_JAVA=OFF \
-D BUILD_opencv_python3=ON \
-D BUILD_opencv_python2=OFF \
-D BUILD_NEW_PYTHON_SUPPORT=ON \
-D OPENCV_SKIP_PYTHON_LOADER=ON \
-D OPENCV_GENERATE_PKGCONFIG=ON \
-D OPENCV_ENABLE_NONFREE=ON \
-D OPENCV_EXTRA_MODULES_PATH=/workspace/opencv/opencv_contrib-4.4.0/modules \
-D WITH_CUDNN=ON \
-D OPENCV_DNN_CUDA=ON \
-D CUDA_ARCH_BIN=6.0 \
-D CUDA_ARCH_PTX=6.0 \
-D CUDNN_LIBRARY=/usr/lib/x86_64-linux-gnu/libcudnn.so.8.0.4 \
-D CUDNN_INCLUDE_DIR=/usr/include ..

RUN make -j16
RUN make install

WORKDIR /workspace

RUN pip install pyhamcrest \
                cython \
                h5py \
                ipykernel \
                matplotlib \ 
                numpy \ 
                statsmodels \
                pandas \
                pillow \
                scipy \
                scikit-image \
                scikit-learn \
                testpath \
                tqdm \
                albumentations \
                timm \
                pytorch_toolbelt \
                tensorboardx

RUN pip install nvidia-pyindex==1.0.5

WORKDIR /workspace

CMD ["/bin/bash"]
