import argparse
import json
import os
from glob import glob
from pathlib import Path

if __name__=="__main__":
    parser = argparse.ArgumentParser("Combine json")
    arg = parser.add_argument
    arg('--result-path', type=str, required=True, help="result file path")
    args = parser.parse_args()

    result = {}
    for json_path in glob(os.path.join(args.result_path, "*/metadata.json")):
        dir = Path(json_path).parent
        with open(json_path, "r") as f:
            jsondata = json.load(f)
            result.update(jsondata)

    with open('output.json','w') as f:
        json.dump(result,f,indent='\t')

    print("Combined Finished : output.json")
