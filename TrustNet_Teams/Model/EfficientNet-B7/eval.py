import argparse
import json
import os
import csv
import math
import numpy as np
from glob import glob
from pathlib import Path
from sklearn.metrics import log_loss

total = 0
correct = 0
true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0

if __name__=="__main__":
    parser = argparse.ArgumentParser("Evaluate Models")
    arg = parser.add_argument
    arg('--fake-threshold', type=float, default=0.5, required=False, help="Fake Threshold")
    arg('--real-threshold', type=float, default=0.5, required=False,
help="Real Threshold")
    arg('--result-path', type=str, required=True, help="result file path")
    arg('--answer-json', type=str, required=False, default="output.json", help="answer json")
    args = parser.parse_args()
    FAKE_thres = args.fake_threshold
    REAL_thres = args.real_threshold
    y = []
    y_pred = []
    with open(args.answer_json) as json_file:
        json_data = json.load(json_file)
        for csv_path in glob(os.path.join(args.result_path, "*.csv")):
            dir = Path(csv_path).parent
            with open(csv_path, "r") as f:
                rdr = csv.reader(f)
                next(rdr)
                for line in rdr:
                    total += 1 
                    json_object = json_data[line[0]]
                    if json_object['label'] == 'FAKE':
                        y.append(1)
                        y_pred.append(float(line[1]))
                        if float(line[1]) >= FAKE_thres:
                            correct += 1
                            true_positive += 1
                        else:
                            false_positive += 1
                    elif json_object['label'] == 'REAL':
                        y.append(0)
                        y_pred.append(float(line[1]))
                        if float(line[1]) < REAL_thres:
                            correct += 1
                            true_negative += 1
                        else:
                            false_negative += 1
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    print('Accuracy \t',correct/total)
    print('Precision\t', precision)
    print('Recall\t\t', recall)
    print('F1 Score\t', 2*(precision * recall) / (precision + recall))
    print('Fall-out\t', false_positive / (true_negative + false_positive))
    print('Log-Loss\t', log_loss(y,y_pred))