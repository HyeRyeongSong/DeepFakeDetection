import argparse
import os
import re
import time
import grpc
import tritonclient.grpc as grpcclient
import torch
import pandas as pd
from trustnet_utils import VideoReader, FaceExtractor, confident_strategy, predict_on_video_set_with_trustnet_api
from training.zoo.myclassifiers import DeepFakeClassifier

done_list = []

if __name__=="__main__":
    parser = argparse.ArgumentParser("Predict test videos")
    arg = parser.add_argument
    arg('--model-name', type=str, required=True, help="TrustNet API Model Name")
    arg('--test-dir', type=str, required=True, help="path to directory with videos")
    arg('--output', type=str, required=False, help="path to output csv", default="submission.csv")
    arg('--size', type=int, required=False, help="input size", default=380)
    arg('--range1', type=int, required=False, help="list(range($range1, $range2))", default=36)
    arg('--range2', type=int, required=False, help="list(range($range1, $range2))", default=50)
    args = parser.parse_args()
    test_list = list(range(args.range1, args.range2))
    try:
        trustnet_client = grpcclient.InferenceServerClient(
            url="localhost:8001",
            verbose=False,
            ssl=False)
    except Exception as e:
        print("channel creation failed: " + str(e))

    for number in test_list:
        if number in done_list: continue
        test_dir = f"{args.test_dir}dfdc_train_part_{number}" # Input path of test datas
        output = f"result/{args.model_name}_{number}.csv"
        print(f"Start inference {test_dir}")
    


        frames_per_video = 32
        video_reader = VideoReader()
        video_read_fn = lambda x: video_reader.read_frames(x, num_frames=frames_per_video)
        face_extractor = FaceExtractor(video_read_fn)
        input_size = args.size
        strategy = confident_strategy
        stime = time.time()

        test_videos = sorted([x for x in os.listdir(test_dir) if x[-4:] == ".mp4"])
        print("Predicting {} videos".format(len(test_videos)))
        predictions = predict_on_video_set_with_trustnet_api(face_extractor=face_extractor, input_size=input_size, trustnet_client=trustnet_client, model_name=args.model_name,
                                           strategy=strategy, frames_per_video=frames_per_video, videos=test_videos,
                                           num_workers=16, test_dir=test_dir)
        print(len(test_videos))
        print(len(predictions))
        submission_df = pd.DataFrame({"filename": test_videos, "label": predictions})
        submission_df.to_csv(output, index=False)
        print("Elapsed:", time.time() - stime)