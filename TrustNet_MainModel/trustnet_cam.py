import torchvision.transforms as transforms
from training.zoo.classifiers import DeepFakeClassifier
import torch
import re
import numpy as np
from trustnet_utils import *
from PIL import Image
from grad_cam import *
batchsize = 32
batch_size = batchsize*4
input_size = 416
frames_per_video = 32
video_reader = VideoReader()
video_read_fn = lambda x: video_reader.read_frames(x, num_frames=frames_per_video)
face_extractor = FaceExtractor(video_read_fn)
faces = face_extractor.process_video("../video.mp4")
print(len(faces))
inputs = []
if len(faces) > 0:
    x = np.zeros((batch_size, input_size, input_size, 3), dtype=np.uint8)
    n = 0
    for frame_data in faces:
        for face in frame_data["faces"]:
            
            resized_face = isotropically_resize_image(face, input_size)
            
            resized_face = put_to_center(resized_face, input_size)
            if n + 1 < 1500:
                x[n] = resized_face
                n += 1
            else:
                pass

    if n > 0:
        raw = x.astype(np.float32)
        x = torch.tensor(x, device="cuda").float()
    # Preprocess the images.
        x = x.permute((0, 3, 1, 2))
        for i in range(len(x)):
            x[i] = normalize_transform(x[i] / 255.)
        x = x[:n]   
        model_path = "weights/best_weight/resnest269rec"
        model = DeepFakeClassifier(encoder="resnest269e")
        checkpoint = torch.load(model_path,map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict({re.sub("^module.", "", k): v for k, v in state_dict.items()}, strict=False)
        model.cuda()


net = model._modules.get("encoder")
grad_cam = GradCam(model=net, feature_module=net.layer4,
                   target_layer_names=["7"], use_cuda=True)
for i in range(len(x)):
    input = x[i].unsqueeze(0)
    path = f"cam/cam_{i}.png"
    target_index = None
    mask = grad_cam(input, target_index)
    show_cam_on_image(raw[i]/255, mask, path)