from .audio_utils import load_audio
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset


class ESC50Dataset(Dataset):
    def __init__(
        self,
        root_dir='/private/home/mandelapatrick/data/ESC_50/audio',
        dataset='esc50',
        metadata_path='/private/home/mandelapatrick/data/ESC_50/esc50.csv',
        val_fold=1,
        mode='train',
        num_samples=10,
        seconds=1,
        random_starts=False,
        nfilter=80
    ):

        # Save input
        self.root_dir = root_dir
        self.nfilter = nfilter

        # Load metadata
        df = pd.read_csv(metadata_path)

        # Get right subset
        if mode not in ['train', 'val']:
            assert("'train' and 'val' are only modes supported")

        # Get train fold
        if mode == 'train':
            df_fold = df[df.fold != val_fold]
            self.filenames = list(df_fold['filename'])
            self.labels = list(df_fold['target'])
        elif mode == 'val':
            df_fold = df[df.fold == val_fold]
            self.filenames = list(df_fold['filename'])
            self.labels = list(df_fold['target'])

        self.dataset = []
        self.seconds = seconds
        self.last_sec = 4.0 if seconds == 1 else 3.0
        if random_starts:
            for i, filename in enumerate(self.filenames):
                count = 0
                for fr in np.random.uniform(0, self.last_sec, num_samples):
                    label = self.labels[i]
                    self.dataset.append((filename, fr, label, i))
                    count += 1
        else:
            for i, filename in enumerate(self.filenames):
                count = 0
                for fr in np.linspace(0, self.last_sec, num_samples):
                    label = self.labels[i]
                    self.dataset.append((filename, fr, label, i))
                    count += 1

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        filename, fr, label, aud_idx = self.dataset[idx]
        filepath = os.path.join(self.root_dir, filename)
        spectogram = load_audio(filepath, fr, 
            num_sec=2, sample_rate=24000, aud_spec_type=2, z_normalize=False)
        return spectogram, label, aud_idx

