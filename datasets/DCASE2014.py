try:
    from audio_utils import load_audio
except:
    from .audio_utils import load_audio
import numpy as np
import pandas as pd
import os
import re
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class DCASEDataset(Dataset):
    def __init__(
        self, 
        root_dir='/private/home/mandelapatrick/data/DCASE2014_mono/', 
        dataset='dcase2014',
        mode='train',
        num_samples=60
    ):

        # Save input
        self.root_dir = root_dir

        # Get right subset
        if mode not in ['train', 'val']:
            assert("'train' and 'val' are only modes supported")
        audio_root_path = os.path.join(root_dir, mode)

        # Create regex string to separate filename into label and count
        r = re.compile("([a-zA-Z]+)([0-9]+)")

        # Label to idx
        label_to_idx = {
            'bus': 0,
            'busystreet': 1,
            'office': 2,
            'openairmarket': 3,
            'park': 4,
            'quietstreet': 5,
            'restaurant': 6,
            'supermarket': 7,
            'tube': 8,
            'tubestation': 9
        }

        self.dataset = []
        for i, filename in enumerate(os.listdir(audio_root_path)):
            full_path = os.path.join(audio_root_path, filename)
            count = 0
            for fr in np.linspace(0, 29.0, num_samples):
                m = r.match(filename).groups()
                label = label_to_idx[m[0]]
                self.dataset.append((full_path, fr, label, i))
                count += 1

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        filename, fr, label, aud_idx = self.dataset[idx]
        filepath = os.path.join(self.root_dir, filename)
        spectogram = load_audio(filepath, fr, num_sec=2, sample_rate=24000, aud_spec_type=2, z_normalize=False)
        if spectogram is None:
            return None
        return spectogram, label, aud_idx


if __name__ == '__main__':

    import time

    val_dataset = DCASEDataset(
        mode='train',
        num_samples=30
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=4, 
        num_workers=0, 
    )
    print(len(val_dataset))

    tic = time.time()
    for idx, batch in enumerate(val_loader):
        if batch is not None:
            audio, label, _ = batch
            print(idx, audio.size(), label, time.time() - tic)
            tic = time.time()
