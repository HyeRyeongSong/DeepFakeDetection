import torch
import grpc
import time
import tritonclient.grpc as grpcclient
from api.deepfake_utils import *
from api.grad_cam import *
from api.classifiers import *

import asyncio

batchsize = 32
batch_size = batchsize * 4

def connectServer(url):
    try:
        triton_client = grpcclient.InferenceServerClient(
            url=url,
            verbose=False,
            ssl=False)
        print("Channel creation success")
    except Exception as e:
        triton_client = None
        print("channel creation failed: " + str(e))

    return triton_client

class Triton:
    def __init__(self, DEBUG=False):
        self.Elapsed = {}
        self.DEBUG = DEBUG
        self.batchsize = 32
        self.batch_size = batchsize * 4
        frames_per_video = 32
        video_reader = VideoReader()
        video_read_fn = lambda t: video_reader.read_frames(t, num_frames=frames_per_video)
        self.face_extractor = FaceExtractor(video_read_fn)
        
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        self.results = []
        self.models = [
            {"model": "416resnest", "name": "resnest", "input_size": 416},
            {"model": "600b7",      "name": "b7",      "input_size": 600}
        ]
        print("Loaded TritonAPI" + (" with DEBUG MODE" if DEBUG else "") )
        self.inputSizes = [ _["input_size"] for _ in self.models ]
        
        model_path = "./api/weights/resnest269rec" # CHKECK
        self.model = DeepFakeClassifier(encoder="resnest269e")
        checkpoint = torch.load(model_path,map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        self.model.load_state_dict({re.sub("^module.", "", k): v for k, v in state_dict.items()}, strict=False)
        self.model.cuda()
        net = self.model._modules.get("encoder")
        self.grad_cam = GradCam(model=net, feature_module=net.layer4,
                           target_layer_names=["7"], use_cuda=True)

    def makeCAM(self, video_name, CAMList):
        raw = self.Saved
        x = torch.tensor(self.x["416"], device="cuda").float()
        for i in CAMList:
            path = f"./app/static/deepfake/{video_name}/cam_{i}.png"
            if not os.path.isdir(f"./app/static/deepfake/{video_name}"):
                os.makedirs(f"./app/static/deepfake/{video_name}")
            mask = self.grad_cam(x[i].unsqueeze(0))
            show_cam_on_image(raw[i], mask, path)
    
    def setDebug(self, boolean):
        self.DEBUG = boolean
        
    def getFaces(self, video):
        return self.face_extractor.process_video(video)
        
    def getResults(self):
        # Only Two Model
        l1 = min(7, len(self.results[0]))
        l2 = min(5, len(self.results[1]))
        l3 = min(3, l1 + l2)
        
        result1 = []
        for idx, j in enumerate(self.results[0]):
            result1.append((j, idx))
        result2 = []
        for idx, j in enumerate(self.results[1]):
            result2.append((j, idx))
        
        # ResNeSt416
        result1 = sorted(result1, reverse=True)[:l1]
        # B7 600
        result2 = sorted(result2, reverse=True)[:l2]
        result  = sorted(result1 + result2, reverse=True)
        percent = 0
        for i in range(l3):
            percent += float(result[i][0])
        percent /= l3
        
        showList = [ i[1] for i in result if i[0] >= 0.81 ]
        showList = set(showList)
        
        return result[l3][0] >= 0.81, percent, showList

    def getX(self, faces, input_size, batch_size):
        inputSize = list(set(input_size))

        x = [ np.zeros((batch_size, inputSize[_], inputSize[_], 3), dtype=np.int8)
                for _ in range(len(inputSize))
        ]
        
        n = 0
        done = False
        for frame_data in faces:
            for face in frame_data["faces"]:
                for k in range(len(inputSize)):
                    resized_face = isotropically_resize_image(face, inputSize[k])
                    resized_face = put_to_center(resized_face, inputSize[k])
                    x[k][n] = resized_face
                n += 1

                if n == batch_size:
                    done = True
                    break
                    
            if done: break
        
        if n > 0:
            self.Saved = x[0].astype(np.float32)
            for k in range(len(inputSize)):
                x[k] = torch.tensor(x[k], device="cuda").float()
                
                x[k] = x[k].permute((0, 3, 1, 2))
                for i in range(len(x[k])):
                    x[k][i] = normalize_transform(x[k][i] / 255.)
                
                x[k] = to_numpy(x[k][:n])
                x[k] = np.ascontiguousarray(x[k])
        else:
            return dict()
        
        return { str(inputSize[k]): x[k] for k in range(len(inputSize)) }
    
    async def request(self, model_name, clientName, x):
        result = list()
        
        inputs  = [ grpcclient.InferInput('input0', x.shape, "FP32") ]
        outputs = [ grpcclient.InferRequestedOutput('output0') ]

        inputs[0].set_data_from_numpy(x)
        
        if self.DEBUG:
            stime = time.time()
        results = await self.loop.run_in_executor(None, lambda: self.tritonclient[clientName].infer(model_name=model_name, \
                                                                                                    inputs=inputs, \
                                                                                                    outputs=outputs))
        
        if self.DEBUG:
            distime = time.time() - stime
            self.Elapsed[model_name] = distime
            print("[Request] Elapsed (After Request) {} {:.3f} second".format(model_name, distime))

        output0_data = results.as_numpy('output0')
        output0_data = sigmoid(output0_data.squeeze())

        return output0_data
    
    async def Detect(self, x):
        if self.DEBUG:
            stime = time.time()
        
        fts = [ asyncio.ensure_future(self.request(self.models[_]["model"], \
                                                   self.models[_]["name"], \
                                                   x[str(self.models[_]["input_size"])])) \
                for _ in range(len(self.models)) ]
        
        self.results = await asyncio.gather(*fts)
        
        if self.DEBUG:
            print("[Async] Elapsed (Before Asnyc) {:.3f} second".format(self.Elapsed['416resnest'] + self.Elapsed['600b7']))
            distime = time.time() - stime
            self.Elapsed['asnyc'] = distime
            print("[Async] Elapsed (After Asnyc) {:.3f} second".format(distime))
            
    def run(self, faces):
        if len(faces) > 0:
            self.tritonclient = {
                "resnest": connectServer("27.96.130.109:8001"),
                "b7":      connectServer("101.101.169.207:8001")
            }
            if self.DEBUG:
                stime = time.time()
            
            self.x = self.getX(faces, input_size=self.inputSizes, batch_size=batch_size)
            if self.DEBUG:
                distime = time.time() - stime
                print("[Preprocessing] Elapsed {:.3f} second".format(distime))
                
            self.loop.run_until_complete(self.Detect(self.x))
            
            if self.DEBUG:
                sync_t = self.Elapsed['416resnest'] + self.Elapsed['600b7']
                async_t = self.Elapsed['asnyc']
                print("[Sync] Total Elapsed Time : {:.3f} second".format(distime + sync_t))
                print("[Async] Total Elapsed Time : {:.3f} second".format(distime + async_t))
