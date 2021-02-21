mkdir torch_weights
mkdir onnx_weights
mkdir tensorrt_weights

# GET Pytorch Model
#wget -O torch_weights/b7_600.pth https://github.com/CryptoSalamander/DeepFake-Detection/releases/download/torchmodel/b7_600
#wget -O torch_weights/resnest269.pth https://github.com/CryptoSalamander/DeepFake-Detection/releases/download/torchmodel/resnest269rec
# GET ONNX Model
#wget -O onnx_weights/b7.onnx https://github.com/CryptoSalamander/DeepFake-Detection/releases/download/onnxmodel/B7_600_BEST.onnx
#wget -O onnx_weights/resnest.onnx https://github.com/CryptoSalamander/DeepFake-Detection/releases/download/onnxmodel/ResNeSt269.onnx
# GET TensorRT Model & Triton Config 
wget -O tensorrt_weights/600b7.zip https://github.com/CryptoSalamander/DeepFake-Detection/releases/download/trustnet_api/600b7.zip
wget -O tensorrt_weights/416resnest.zip https://github.com/CryptoSalamander/DeepFake-Detection/releases/download/trustnet_api/416resnest.zip

unzip tensorrt_weights/600b7.zip -d tensorrt_weights/600b7
unzip tensorrt_weights/416resnest.zip -d  tensorrt_weights//416resnest

rm -rf tensorrt_weights/*.zip