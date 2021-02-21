import tensorrt as trt
import torch
import pycuda.driver as cuda
import pycuda.autoinit
import os
import time
import pandas as pd
import argparse
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
from trustnet_utils import *
import numpy as np
parser = argparse.ArgumentParser("Predict test videos")
arg = parser.add_argument
arg('--models', type=str, required=True, help="checkpoint files")
arg('--test-dir', type=str, required=True, help="path to directory with videos")
arg('--output', type=str, required=False, help="path to output csv", default="submission.csv")
arg('--gpu', type=str, required=False, help="decide gpu device", default="0")
#arg('--size', type=int, required=False, help="input size", default=380)
#arg('--range1', type=int, required=False, help="list(range($range1, $range2))", default=36)
#arg('--range2', type=int, required=False, help="list(range($range1, $range2))", default=50)
args = parser.parse_args()
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
def sigmoid(x):
    return 1 / (1 +np.exp(-x))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
batchsize = 32
batch_size = batchsize*4
input_size = 600
frames_per_video = 32
video_reader = VideoReader()
video_read_fn = lambda x: video_reader.read_frames(x, num_frames=frames_per_video)
face_extractor = FaceExtractor(video_read_fn)
test_videos = sorted([x for x in os.listdir(args.test_dir) if x[-4:] == ".mp4"])
total_stime = time.time()
with open(args.models, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    results = []
    stream = cuda.Stream()
    engine = runtime.deserialize_cuda_engine(f.read())
    print("Engine Loaded Successfully")
    with engine.create_execution_context() as context:
        for test_video in test_videos:
            stime = time.time()
            faces = face_extractor.process_video(args.test_dir+test_video)
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
                    #x = torch.tensor(x).float().cuda()
                    x = torch.tensor(x).float()

                    # Preprocess the images.
                    x = x.permute((0, 3, 1, 2))
                    for i in range(len(x)):
                        x[i] = normalize_transform(x[i] / 255.)
                    x = x[:n]
                    print(x.shape)
                    #x = to_numpy(x)
                    x = np.ascontiguousarray(x)
                    print("Get X Time : ", time.time() - stime)
                    output = np.empty(x.shape[0], dtype=np.float32)
                    d_input = cuda.mem_alloc(1 * x.nbytes)
                    d_output = cuda.mem_alloc(1 * output.nbytes)
                    bindings = [int(d_input), int(d_output)]

                    context.get_binding_shape(0)
                    context.set_binding_shape(0, x.shape) #x.shape)
                    context.get_binding_shape(0)
                    cuda.memcpy_htod_async(d_input, x, stream)
                    context.execute_async_v2(bindings= bindings, stream_handle=stream.handle)
                    cuda.memcpy_dtoh_async(output, d_output, stream)
                    stream.synchronize()
                    output = sigmoid(output.squeeze())
                    output = np.mean(output)
                    print(output)
                    results.append(output)
                    

print("Elapsed:", time.time() - total_stime)
submission_df = pd.DataFrame({"filename": test_videos, "label": results})
submission_df.to_csv(args.output, index=False)