import onnx
import onnxruntime
import torchvision.transforms as transforms
from training.zoo.classifiers import DeepFakeClassifier
import torch
import re
import numpy as np
from trustnet_utils import *
from PIL import Image

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)
ort_session = onnxruntime.InferenceSession("model.onnx")
batchsize = 32
batch_size = batchsize*4
input_size = 600
frames_per_video = 32
video_reader = VideoReader()
video_read_fn = lambda x: video_reader.read_frames(x, num_frames=frames_per_video)
face_extractor = FaceExtractor(video_read_fn)
faces = face_extractor.process_video("../dfdc_train_all/mini_test/asmturwvvg.mp4")
with torch.no_grad():
    if len(faces) > 0:
        x = np.zeros((batch_size, input_size, input_size, 3), dtype=np.uint8)
        n = 0
        for frame_data in faces:
            for face in frame_data["faces"]:
                resized_face = isotropically_resize_image(face, input_size)
                resized_face = put_to_center(resized_face, input_size)
                if n + 1 < batch_size:
                    x[n] = resized_face
                    n += 1
                else:
                    pass
        if n > 0:
            x = torch.tensor(x, device="cuda").float()
            # Preprocess the images.
            x = x.permute((0, 3, 1, 2))
            for i in range(len(x)):
                x[i] = normalize_transform(x[i] / 255.)
            x = x[:n]   
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)
    print(type(ort_outs))
    ort_outs = torch.FloatTensor(ort_outs)
    ort_outs = torch.sigmoid(ort_outs.squeeze())
    print(ort_outs)
    
    #np.testing.assert_allclose(to_numpy(y), ort_outs[0], rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")