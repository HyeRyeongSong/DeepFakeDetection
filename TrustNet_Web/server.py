#### Check Pretrained
import os

PRETRAINED_DIR = "./api/weights/resnest269rec"
DOWNLOAD_URL   = "https://github.com/CryptoSalamander/DeepFake-Detection/releases/download/torchmodel/resnest269rec"

if not os.path.isfile(PRETRAINED_DIR):
    print(f"DOWNLOAD pretrained model -> {PRETRAINED_DIR}")
    os.system(f"wget -O {PRETRAINED_DIR} {DOWNLOAD_URL}")
####

from app import app, db
from app.models import User, Video, Likes, Comments
import argparse 

@app.shell_context_processor
def make_shell_context():
    return {'db': db, 'User': User, 'Video': Video, 'Likes': Likes, 'Comments': Comments}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default="8003", help='PORT')
    parser.add_argument('--ip', type=str, default='0.0.0.0', help='IP')
    parser.add_argument('--debug', action='store_true', default=False, help="DEUBG MODE")
    args = parser.parse_args()

    app.run(debug=args.debug, host=args.ip, port=args.port)
