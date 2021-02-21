import torch
import grpc
import tritonclient.grpc as grpcclient
from trustnet_utils import *
batchsize = 32
batch_size = batchsize*4
input_size = 600
frames_per_video = 32
video_reader = VideoReader()
video_read_fn = lambda x: video_reader.read_frames(x, num_frames=frames_per_video)
face_extractor = FaceExtractor(video_read_fn)
test_videos = sorted([x for x in os.listdir("../dfdc_train_all/mini_test") if x[-4:] == ".mp4"])
stime = time.time()
faces = face_extractor.process_video("../dfdc_train_all/mini_test/asmturwvvg.mp4")
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
    rawx = x
    #x = torch.tensor(x).float().cuda()
    x = torch.tensor(x).float()

    # Preprocess the images.
    x = x.permute((0, 3, 1, 2))
    for i in range(len(x)):
        x[i] = normalize_transform(x[i] / 255.)
    x = x[:n]
    print("Elapsed Get X : ", time.time() - stime)
    print(x.shape)
    #x = to_numpy(x)
    npx = np.ascontiguousarray(x)

try:
    triton_client = grpcclient.InferenceServerClient(
        url="localhost:8001",
        verbose=False,
        ssl=False)
except Exception as e:
    print("channel creation failed: " + str(e))

model_name = "b7fp16"

inputs = []
outputs = []
inputs.append(grpcclient.InferInput('input0', [66, 3, 600, 600], "FP32"))
input0_data = x

inputs[0].set_data_from_numpy(input0_data)
outputs.append(grpcclient.InferRequestedOutput('output0'))

stime = time.time()
results = triton_client.infer(model_name=model_name, inputs=inputs, outputs=outputs)

inputs[0].set_data_from_numpy(input0_data)
print('Elapsed : ', time.time() - stime)
output0_data = results.as_numpy('output0')

def sigmoid(x):
    return 1 / (1 +np.exp(-x))

output0_data = sigmoid(output0_data.squeeze())
print(output0_data)