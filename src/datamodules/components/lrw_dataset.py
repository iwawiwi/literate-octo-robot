import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from turbojpeg import TJPF_GRAY, TurboJPEG

from src.utils.image_transform import CenterCrop, HorizontalFlip, RandomCrop

jpeg = TurboJPEG()


class LRWDataset(Dataset):
    def __init__(self, root_dir, phase):

        with open(os.path.join(root_dir, "label_sorted.txt")) as myfile:
            self.labels = myfile.read().splitlines()

        self.list = []
        self.unlabel_list = []
        self.phase = phase

        # if not hasattr(self.args, "is_aug"):
        #     setattr(self.args, "is_aug", True)  # FIXME: is it necessary?

        for (i, label) in enumerate(self.labels):
            files = glob.glob(os.path.join(root_dir, label, phase, "*.pkl"))
            files = sorted(files)

            self.list += [file for file in files]

    def __getitem__(self, idx):

        tensor = torch.load(self.list[idx])

        inputs = tensor.get("video")
        inputs = [jpeg.decode(img, pixel_format=TJPF_GRAY) for img in inputs]
        inputs = np.stack(inputs, 0) / 255.0
        inputs = inputs[:, :, :, 0]

        if self.phase == "train":
            batch_img = RandomCrop(inputs, (88, 88))
            batch_img = HorizontalFlip(batch_img)
        elif self.phase == "val" or self.phase == "test":
            batch_img = CenterCrop(inputs, (88, 88))

        result = {}
        result["video"] = torch.FloatTensor(batch_img[:, np.newaxis, ...])
        # print(result['video'].size())
        result["label"] = tensor.get("label")
        result["duration"] = 1.0 * tensor.get("duration")

        return result

    def __len__(self):
        return len(self.list)

    def load_duration(self, file):
        with open(file) as f:
            lines = f.readlines()
            for line in lines:
                if line.find("Duration") != -1:
                    duration = float(line.split(" ")[1])

        tensor = torch.zeros(29)
        mid = 29 / 2
        start = int(mid - duration / 2 * 25)
        end = int(mid + duration / 2 * 25)
        tensor[start:end] = 1.0
        return tensor


class LRW1000_Dataset(Dataset):
    def __init__(self, root_dir, phase):

        self.data = []
        self.phase = phase
        if self.phase == "train":
            self.index_root = "LRW1000_Public_pkl_jpeg/trn"
        else:
            self.index_root = "LRW1000_Public_pkl_jpeg/tst"

        self.data = glob.glob(os.path.join(root_dir, self.index_root, "*.pkl"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        pkl = torch.load(self.data[idx])
        video = pkl.get("video")
        video = [jpeg.decode(img, pixel_format=TJPF_GRAY) for img in video]
        video = np.stack(video, 0)
        video = video[:, :, :, 0]

        if self.phase == "train":
            video = RandomCrop(video, (88, 88))
            video = HorizontalFlip(video)
        elif self.phase == "val" or self.phase == "test":
            video = CenterCrop(video, (88, 88))

        pkl["video"] = torch.FloatTensor(video)[:, None, ...] / 255.0

        return pkl
