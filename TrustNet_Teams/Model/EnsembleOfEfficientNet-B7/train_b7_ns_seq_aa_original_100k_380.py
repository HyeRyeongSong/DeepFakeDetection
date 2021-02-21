import yaml
import os
import random
import tqdm

import numpy as np
from PIL import Image

import torch
from torch import nn
from torch import distributions
from torch.nn import functional as F
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

import ffmpeg

from albumentations import ImageOnlyTransform
from albumentations import SmallestMaxSize, HorizontalFlip, Normalize, Compose, RandomCrop
from albumentations.pytorch import ToTensor
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.model import MBConvBlock

from timm.data.transforms_factory import transforms_imagenet_train

from datasets import TrackPairDataset
from extract_tracks_from_videos import TRACK_LENGTH, TRACKS_ROOT
from generate_track_pairs import TRACK_PAIRS_FILE_NAME

SEED = 20
BATCH_SIZE = 2 #8
TRAIN_INDICES = [19, 21, 23, 25, 27, 29, 31]
INITIAL_LR = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 16
NUM_WARMUP_ITERATIONS = 100
SNAPSHOT_FREQUENCY = 4000 #1000
OUTPUT_FOLDER_NAME = 'efficientnet-b7_ns_seq_aa-original-mstd0.5_100k_380'
SNAPSHOT_NAME_TEMPLATE = 'snapshot_{}.pth'
FINAL_SNAPSHOT_NAME = 'final.pth'
MAX_ITERS = 400000 #100000

FPS_RANGE = (15, 30)
SCALE_RANGE = (0.25, 1)
CRF_RANGE = (17, 40)
TUNE_VALUES = ['film', 'animation', 'grain', 'stillimage', 'fastdecode', 'zerolatency']

MIN_SIZE = 380
CROP_HEIGHT = 380
CROP_WIDTH = 380

PRETRAINED_WEIGHTS_PATH = 'external_data/noisy_student_efficientnet-b7.pth'
SNAPSHOTS_ROOT = 'snapshots'
LOGS_ROOT = 'logs'

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='1'

class SeqExpandConv(nn.Module):
    def __init__(self, in_channels, out_channels, seq_length):
        super(SeqExpandConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False)
        self.seq_length = seq_length

    def forward(self, x):
        batch_size, in_channels, height, width = x.shape
        x = x.view(batch_size // self.seq_length, self.seq_length, in_channels, height, width)
        x = self.conv(x.transpose(1, 2).contiguous()).transpose(2, 1).contiguous()
        x = x.flatten(0, 1)
        return x


class TrackTransform(object):
    def __init__(self, fps_range, scale_range, crf_range, tune_values):
        self.fps_range = fps_range
        self.scale_range = scale_range
        self.crf_range = crf_range
        self.tune_values = tune_values

    def get_params(self, src_fps, src_height, src_width):
        if random.random() > 0.5:
            return None

        dst_fps = src_fps
        if random.random() > 0.5:
            dst_fps = random.randrange(*self.fps_range)

        scale = 1.0
        if random.random() > 0.5:
            scale = random.uniform(*self.scale_range)

        dst_height = round(scale * src_height) // 2 * 2
        dst_width = round(scale * src_width) // 2 * 2

        crf = random.randrange(*self.crf_range)
        tune = random.choice(self.tune_values)

        return dst_fps, dst_height, dst_width, crf, tune

    def __call__(self, track_path, src_fps, dst_fps, dst_height, dst_width, crf, tune):
        out, err = (
            ffmpeg
                .input(os.path.join(track_path, '%d.png'), framerate=src_fps, start_number=0)
                .filter('fps', fps=dst_fps)
                .filter('scale', dst_width, dst_height)
                .output('pipe:', format='h264', vcodec='libx264', crf=crf, tune=tune)
                .run(capture_stdout=True, quiet=True)
        )
        out, err = (
            ffmpeg
                .input('pipe:', format='h264')
                .output('pipe:', format='rawvideo', pix_fmt='rgb24')
                .run(capture_stdout=True, input=out, quiet=True)
        )

        imgs = np.frombuffer(out, dtype=np.uint8).reshape(-1, dst_height, dst_width, 3)

        return imgs


class VisionTransform(ImageOnlyTransform):
    def __init__(
            self, transform, always_apply=False, p=1.0
    ):
        super(VisionTransform, self).__init__(always_apply, p)
        self.transform = transform

    def apply(self, image, **params):
        return np.array(self.transform(Image.fromarray(image)))

    def get_transform_init_args_names(self):
        return ("transform")


def set_global_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def prepare_cudnn(deterministic=None, benchmark=None):
    # https://pytorch.org/docs/stable/notes/randomness.html#cudnn
    if deterministic is None:
        deterministic = os.environ.get("CUDNN_DETERMINISTIC", "True") == "True"
    torch.backends.cudnn.deterministic = deterministic

    # https://discuss.pytorch.org/t/how-should-i-disable-using-cudnn-in-my-code/38053/4
    if benchmark is None:
        benchmark = os.environ.get("CUDNN_BENCHMARK", "True") == "True"
    torch.backends.cudnn.benchmark = benchmark


def main():
    with open('config.yaml', 'r') as f:
        config = yaml.load(f)

    set_global_seed(SEED)
    prepare_cudnn(deterministic=True, benchmark=True)

    model = EfficientNet.from_name('efficientnet-b7', override_params={'num_classes': 1})
    state = torch.load(PRETRAINED_WEIGHTS_PATH, map_location=lambda storage, loc: storage)
    state.pop('_fc.weight')
    state.pop('_fc.bias')
    res = model.load_state_dict(state, strict=False)
    assert set(res.missing_keys) == set(['_fc.weight', '_fc.bias']), 'issue loading pretrained weights'

    for module in model.modules():
        if isinstance(module, MBConvBlock):
            if module._block_args.expand_ratio != 1:
                expand_conv = module._expand_conv
                seq_expand_conv = SeqExpandConv(expand_conv.in_channels, expand_conv.out_channels, len(TRAIN_INDICES))
                seq_expand_conv.conv.weight.data[:, :, 0, :, :].copy_(expand_conv.weight.data / 3)
                seq_expand_conv.conv.weight.data[:, :, 1, :, :].copy_(expand_conv.weight.data / 3)
                seq_expand_conv.conv.weight.data[:, :, 2, :, :].copy_(expand_conv.weight.data / 3)
                module._expand_conv = seq_expand_conv

    model = model.cuda()

    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    _, rand_augment, _ = transforms_imagenet_train((CROP_HEIGHT, CROP_WIDTH), auto_augment='original-mstd0.5',
                                                   separate=True)

    train_dataset = TrackPairDataset(os.path.join(config['ARTIFACTS_PATH'], TRACKS_ROOT),
                                     os.path.join(config['ARTIFACTS_PATH'], TRACK_PAIRS_FILE_NAME),
                                     TRAIN_INDICES,
                                     track_length=TRACK_LENGTH,
                                     track_transform=TrackTransform(FPS_RANGE, SCALE_RANGE, CRF_RANGE, TUNE_VALUES),
                                     image_transform=Compose([
                                         SmallestMaxSize(MIN_SIZE),
                                         HorizontalFlip(),
                                         RandomCrop(CROP_HEIGHT, CROP_WIDTH),
                                         VisionTransform(rand_augment, p=0.5),
                                         normalize,
                                         ToTensor()
                                     ]), sequence_mode=True)

    print('Train dataset size: {}.'.format(len(train_dataset)))

    warmup_optimizer = torch.optim.SGD(model._fc.parameters(), INITIAL_LR, momentum=MOMENTUM,
                                       weight_decay=WEIGHT_DECAY, nesterov=True)

    full_optimizer = torch.optim.SGD(model.parameters(), INITIAL_LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY,
                                     nesterov=True)
    full_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(full_optimizer,
                                                          lambda iteration: (MAX_ITERS - iteration) / MAX_ITERS)

    snapshots_root = os.path.join(config['ARTIFACTS_PATH'], SNAPSHOTS_ROOT, OUTPUT_FOLDER_NAME)
    os.makedirs(snapshots_root)
    log_root = os.path.join(config['ARTIFACTS_PATH'], LOGS_ROOT, OUTPUT_FOLDER_NAME)
    os.makedirs(log_root)

    writer = SummaryWriter(log_root)

    iteration = 0
    if iteration < NUM_WARMUP_ITERATIONS:
        print('Start {} warmup iterations'.format(NUM_WARMUP_ITERATIONS))
        model.eval()
        model._fc.train()
        for param in model.parameters():
            param.requires_grad = False
        for param in model._fc.parameters():
            param.requires_grad = True
        optimizer = warmup_optimizer
    else:
        print('Start without warmup iterations')
        model.train()
        optimizer = full_optimizer

    max_lr = max(param_group["lr"] for param_group in full_optimizer.param_groups)
    writer.add_scalar('train/max_lr', max_lr, iteration)

    epoch = 0
    fake_prob_dist = distributions.beta.Beta(0.5, 0.5)
    while True:
        epoch += 1
        print('Epoch {} is in progress'.format(epoch))
        loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
        for samples in tqdm.tqdm(loader):
            iteration += 1
            fake_input_tensor = torch.stack(samples['fake']).transpose(0, 1).cuda()
            real_input_tensor = torch.stack(samples['real']).transpose(0, 1).cuda()
            target_fake_prob = fake_prob_dist.sample((len(fake_input_tensor),)).float().cuda()
            fake_weight = target_fake_prob.view(-1, 1, 1, 1, 1)

            input_tensor = (1.0 - fake_weight) * real_input_tensor + fake_weight * fake_input_tensor
            pred = model(input_tensor.flatten(0, 1)).flatten()

            loss = F.binary_cross_entropy_with_logits(pred, target_fake_prob.repeat_interleave(len(TRAIN_INDICES)))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if iteration > NUM_WARMUP_ITERATIONS:
                full_lr_scheduler.step()
                max_lr = max(param_group["lr"] for param_group in full_optimizer.param_groups)
                writer.add_scalar('train/max_lr', max_lr, iteration)

            writer.add_scalar('train/loss', loss.item(), iteration)

            if iteration == NUM_WARMUP_ITERATIONS:
                print('Stop warmup iterations')
                model.train()
                for param in model.parameters():
                    param.requires_grad = True
                optimizer = full_optimizer

            if iteration % SNAPSHOT_FREQUENCY == 0:
                snapshot_name = SNAPSHOT_NAME_TEMPLATE.format(iteration)
                snapshot_path = os.path.join(snapshots_root, snapshot_name)
                print('Saving snapshot to {}'.format(snapshot_path))
                torch.save(model.state_dict(), snapshot_path)

            if iteration >= MAX_ITERS:
                print('Stop training due to maximum iteration exceeded')
                return


if __name__ == '__main__':
    main()
