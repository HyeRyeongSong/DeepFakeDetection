import torch
import geffnet
from torch2trt import torch2trt
import re
from training.zoo.classifiers import DeepFakeClassifier
from trustnet_utils import *
import onnx
print(torch.__version__)

x = torch.randn((128, 3, 600, 600), requires_grad=True)
with torch.no_grad():
    model_path = "weights/model.pth"
    model = DeepFakeClassifier(encoder="tf_efficientnet_b7_ns")
    checkpoint = torch.load("weights/b7_600/b7_600_DeepFakeClassifier_tf_efficientnet_b7_ns_0_best_dice",map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict({re.sub("^module.", "", k): v for k, v in state_dict.items()}, strict=False)
    print("Model's state_dict:")
    model.eval()
    model(x)
    print("==> Exporting model to ONNX format at '{}'".format("model.onnx"))
    input_names = ["input0"]
    output_names = ["output0"]
    dynamic_axes = {'input0': {0: 'batch'}, 'output0': {0: 'batch'}}
    export_type = torch.onnx.OperatorExportTypes.ONNX
    torch_out = torch.onnx._export(
            model, x, "model.onnx", export_params=True, verbose=True, input_names=input_names,
            output_names=output_names, keep_initializers_as_inputs=False, dynamic_axes=dynamic_axes,
            opset_version=10, operator_export_type=export_type)

    print("==> Loading and checking exported model from '{}'".format("model.onnx"))
    onnx_model = onnx.load("model.onnx")
    onnx.checker.check_model(onnx_model)  # assuming throw on error
    print("==> Passed")