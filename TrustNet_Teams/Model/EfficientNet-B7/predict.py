import argparse
import os
import re
import time

import torch
import pandas as pd
from kernel_utils import VideoReader, FaceExtractor, confident_strategy, predict_on_video_set
from training.zoo.classifiers import DeepFakeClassifier


done_list = []

if __name__=="__main__":
    parser = argparse.ArgumentParser("Predict test videos")
    arg = parser.add_argument
    arg('--weights-dir', type=str, default="weights", help="path to directory with checkpoints")
    arg('--models', nargs='+', required=True, help="checkpoint files")
    arg('--test-dir', type=str, required=True, help="path to directory with videos")
    arg('--output', type=str, required=False, help="path to output csv", default="submission.csv")
    arg('--encoder', type=str, required=True, help="encoder", default="resnest269e")
    arg('--input-size', type=int, required=True, help="input size", default=320)
    arg('--num-workers', type=int, required=True, help="cpu cores", default=1)
    arg('--range1', type=int, required=False, help="test number($range1~$range2)", default=36)
    arg('--range2', type=int, required=False, help="test number($range1~$range2)", defulat=50)
    args = parser.parse_args()
    test_list = list(range(args.range1, args.range2))

    for number in test_list:
        if number in done_list: continue
        test_dir = f"{args.test_dir}dfdc_train_part_{number}" # Input path of test datas
        output = f"{number}.csv"
        print(f"Start inference {test_dir}")
    
        models = []
        model_paths = [os.path.join(args.weights_dir, model) for model in args.models]
        for path in model_paths:
            model = DeepFakeClassifier(encoder=args.encoder).to("cuda")
            print("loading state dict {}".format(path))
            checkpoint = torch.load(path, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint)
            model.load_state_dict({re.sub("^module.", "", k): v for k, v in state_dict.items()}, strict=False)
            model.eval()
            del checkpoint
            models.append(model.half())

        frames_per_video = 32
        video_reader = VideoReader()
        video_read_fn = lambda x: video_reader.read_frames(x, num_frames=frames_per_video)
        face_extractor = FaceExtractor(video_read_fn)
        strategy = confident_strategy
        stime = time.time()

        test_videos = sorted([x for x in os.listdir(test_dir) if x[-4:] == ".mp4"])
        print("Predicting {} videos".format(len(test_videos)))
        predictions = predict_on_video_set(face_extractor=face_extractor, input_size=args.input_size, models=models,
                                           strategy=strategy, frames_per_video=frames_per_video, videos=test_videos,
                                           num_workers=args.num_workers, test_dir=test_dir)
        submission_df = pd.DataFrame({"filename": test_videos, "label": predictions})
        submission_df.to_csv(output, index=False)
        print("Elapsed:", time.time() - stime)

# python predict_folder.py --models "b7_888_DeepFakeClassifier_resnest269e_0_37" --test-dir "/workspace/dataset/test/" --encoder "resnest269e" --input_size 380 --num_workers 16
