#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import av
import ffmpeg
from joblib import Parallel, delayed
from multiprocessing import Manager
import os
import pickle
import random
import torch
import torchvision
import torch.utils.data
import glob

try: 
    from decoder import decode
except:
    from .decoder import decode

try: 
    from video_transforms import (
        random_short_side_scale_jitter, 
        random_crop, 
        horizontal_flip, 
        grayscale, 
        color_jitter, 
        uniform_crop, 
        resize, 
        normalize
    )
except:
    from .video_transforms import (
        random_short_side_scale_jitter, 
        random_crop, 
        horizontal_flip, 
        grayscale, 
        color_jitter, 
        uniform_crop, 
        resize, 
        normalize
    )


# Enable multi thread decoding.
ENABLE_MULTI_THREAD_DECODE = True

# Decoding backend, options include `pyav` or `torchvision`
DECODING_BACKEND = 'pyav'

MEAN=[0.45, 0.45, 0.45]
STD=[0.225, 0.225, 0.225]

ROOT_DIR = {
    'kinetics': '/datasets01/kinetics/070618/',
    'kinetics600': '/datasets01/kinetics/070618/600/',
}

MODE_DIR = {
    'kinetics': {
        'train': 'train_avi-480p',
        'val': 'val_avi-480p'
    },
    'kinetics600': {
        'train': 'train',
        'val': 'val'
    },
}


def valid_video(vid_idx, vid_path):
    try:
        probe = ffmpeg.probe(vid_path)
        video_stream = next((
            stream for stream in probe['streams'] if stream['codec_type'] == 'video'), 
            None
        )
        audio_stream = next((
            stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), 
            None
        )
        if audio_stream and video_stream and float(video_stream['duration']) > 1.1 and float(audio_stream['duration']) > 1.1:
            print(f"{vid_idx}: True", flush=True)
            return True
        else:
            print(f"{vid_idx}: False (duration short/ no audio)", flush=True)
            return False
    except:
        print(f"{vid_idx}: False", flush=True)
        return False


def filter_videos(vid_paths):
    all_indices = Parallel(n_jobs=30)(delayed(valid_video)(vid_idx, vid_paths[vid_idx]) for vid_idx in range(len(vid_paths)))
    valid_indices = [i for i, val in enumerate(all_indices) if val]
    return valid_indices


def get_video_container(path_to_vid, multi_thread_decode=False, backend="pyav"):
    """
    Given the path to the video, return the pyav video container.
    Args:
        path_to_vid (str): path to the video.
        multi_thread_decode (bool): if True, perform multi-thread decoding.
        backend (str): decoder backend, options include `pyav` and
            `torchvision`, default is `pyav`.
    Returns:
        container (container): video container.
    """
    if backend == "torchvision":
        with open(path_to_vid, "rb") as fp:
            container = fp.read()
        return container
    elif backend == "pyav":
        try:
            container = av.open(path_to_vid)
        except:
            container = av.open(path_to_vid, metadata_errors="ignore")
        if multi_thread_decode:
            # Enable multiple threads for decoding.
            container.streams.video[0].thread_type = "AUTO"
        return container
    else:
        raise NotImplementedError("Unknown backend {}".format(backend))


class GDTPretrainDataset(torch.utils.data.Dataset):
    """
    Audio-video loader. Construct the video loader, then sample
    clips from the videos. For training and validation, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(
        self,
        ds_name='kinetics',
        mode='train',
        root_dir=None,
        args=None,
    ):
        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for {}".format(mode, ds_name)
        self.ds_name = ds_name
        self.mode = mode
        self.num_frames = args.clip_len
        self.sample_rate = args.sample_rate
        self.train_crop_size = args.train_crop_size
        if self.train_crop_size in [112, 128]:
            train_jitter_scles = (128, 160)
        else:
            train_jitter_scles = (256, 320)
        self.train_jitter_scles = train_jitter_scles
        self.num_ensemble_views = 1
        self.num_spatial_crops = 1
        if root_dir is None:
            self.data_prefix = os.path.join(ROOT_DIR[ds_name], MODE_DIR[ds_name][mode])
        else:
            self.data_prefix = root_dir
        self.path_to_data_dir = 'datasets/data'
        self.num_data_samples = args.num_data_samples
        self.colorjitter = args.colorjitter
        self.sync = False #args.asynced == 0
        self.target_fps = args.target_fps
        self.decode_audio = args.decode_audio
        self.aug_audio = args.aug_audio
        self.num_sec = args.num_sec
        self.aud_sample_rate = args.aud_sample_rate
        self.aud_spec_type = args.aud_spec_type
        self.use_volume_jittering = args.use_volume_jittering
        self.use_temporal_jittering = args.use_temporal_jittering
        self.z_normalize = args.z_normalize

        self._video_meta = {}

        # Get classes
        if self.ds_name != 'audioset':
            classes = list(sorted(glob.glob(os.path.join(self.data_prefix, '*'))))
            classes = [os.path.basename(i) for i in classes]
            self.class_to_idx = {classes[i]: i for i in range(len(classes))}

        # For training or validation mode, one single clip is sampled from every video.
        # For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every video.
        # For every clip, NUM_SPATIAL_CROPS is cropped spatially from the frames.
        if self.mode in ["train", "val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = (
                self.num_ensemble_views * self.num_spatial_crops
            )

        # self.manager = Manager()
        print(f"Constructing {self.ds_name} {self.mode}...")
        self._construct_loader()

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        # Get list of paths
        os.makedirs(self.path_to_data_dir, exist_ok=True)
        path_to_file = os.path.join(
            self.path_to_data_dir, f"{self.ds_name}_{self.mode}.txt"
        )
        if not os.path.exists(path_to_file) and self.ds_name != 'audioset':
            files = list(sorted(glob.glob(os.path.join(self.data_prefix, '*', '*')))) 
            with open(path_to_file, 'w') as f:
                for item in files:
                    f.write("%s\n" % item)

        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []
        self._vid_indices = []
        with open(path_to_file, "r") as f:
            for clip_idx, path in enumerate(f.read().splitlines()):
                for idx in range(self._num_clips):
                    self._path_to_videos.append(
                        os.path.join(self.data_prefix, path)
                    )
                    if self.ds_name != 'audioset':
                        class_name = path.split('/')[-2]
                        label = self.class_to_idx[class_name]
                    self._labels.append(int(label))
                    self._spatial_temporal_idx.append(idx)
                    self._vid_indices.append(clip_idx)
                    self._video_meta[clip_idx * self._num_clips + idx] = {}
        assert (
            len(self._path_to_videos) > 0
        ), "Failed to load {} split {} from {}".format(
            self.ds_name, self._split_idx, path_to_file
        )
        print(
            "Constructing {} dataloader (size: {}) from {}".format(
                self.ds_name, len(self._path_to_videos), path_to_file
            )
        )

        # Create / Load valid indices (has audio)
        vid_valid_file = f'{self.path_to_data_dir}/{self.ds_name}_valid.pkl'
        if os.path.exists(vid_valid_file):
            with open(vid_valid_file, 'rb') as handle:
                self.valid_indices = pickle.load(handle)
        else:
            self.valid_indices = filter_videos(self._path_to_videos)
            with open(vid_valid_file, 'wb') as handle:
                pickle.dump(
                    self.valid_indices, 
                    handle, 
                    protocol=pickle.HIGHEST_PROTOCOL
                )
        if self.num_data_samples is not None:
            self.valid_indices = self.valid_indices[:self.num_data_samples]
        print(f"Total number of videos: {len(self._path_to_videos)}, Valid videos: {len(self.valid_indices)}", flush=True)

        # Make lists a Manager objects
        #self._path_to_videos = self.manager.list(self._path_to_videos)
        self.valid_indices = list(self.valid_indices)

    def __getitem__(self, index):
        """
        Given the video index, return tensors: video, audio, label, vid_idx, idx
        Otherwise, repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        # T1: Sample selection
        index_capped = index
        index = self.valid_indices[index_capped]

        # Get two random shifts
        clip_idx1 = random.randint(0, 1000)
        clip_idx2 = clip_idx1 if self.sync else random.randint(0, 1000)
        clip_idx2 = clip_idx2 + 1000 if clip_idx2 < 0 else clip_idx2
        num_clips = 1000
        clip_idx_list = [clip_idx1, clip_idx2]
            
        # Lists to store GDTs
        V = []
        A = []

        # T2: Temporal Shift transformation: tau_1, tau_2
        for tau_ix in range(2):

            time_idx = clip_idx_list[tau_ix]

            # Get video container
            video_container = get_video_container(
                self._path_to_videos[index],
                ENABLE_MULTI_THREAD_DECODE,
                DECODING_BACKEND,
            )
            
            # T3: Modality splicing transformation: (V, A)
            frames, spec = decode(
                self._path_to_videos[index],
                video_container,
                self.sample_rate,
                self.num_frames,
                time_idx,
                num_clips=num_clips,
                video_meta=self._video_meta[index],
                target_fps=self.target_fps,
                backend=DECODING_BACKEND,
                max_spatial_scale=self.train_jitter_scles[1],
                decode_audio=self.decode_audio,
                aug_audio=self.aug_audio,
                num_sec=self.num_sec,
                aud_sample_rate=self.aud_sample_rate,
                aud_spec_type=self.aud_spec_type,
                use_volume_jittering=self.use_volume_jittering,
                use_temporal_jittering=self.use_temporal_jittering,
                z_normalize=self.z_normalize,
            )

            # T4: Time Reversal Operation: (R, RT)
            for r_ix in range(2):

                # Clone frames and spec
                no_aug_frames = frames.clone()
                aug_spec = spec.clone()

                # Reverse audio and video
                if r_ix % 2 == 0:
                    no_aug_frames = no_aug_frames
                    aug_spec = aug_spec
                else: 
                    no_aug_frames = no_aug_frames.flip(0) # T H W C
                    aug_spec = aug_spec.flip(-1) # F x T
                        
                # T5: Data Augmentation: (gv, ga)
                aug_frames = self.augmentation(no_aug_frames)

                # Add to V, A list
                V.append(aug_frames)
                A.append(aug_spec)

        label = self._labels[index]
        vid_idx = self._vid_indices[index]	
        idx = index

        return torch.cat(V, dim=0), torch.cat(A, dim=0), label, vid_idx, index_capped

    def augmentation(self, frames):
        # Crop params
        spatial_sample_index = -1
        min_scale = self.train_jitter_scles[0]
        max_scale = self.train_jitter_scles[1]
        crop_size = self.train_crop_size

        # Normalization
        frames = frames.float()
        frames = frames / 255.0

        # T H W C -> C T H W.
        frames = frames.permute(3, 0, 1, 2)
        
        # Perform data augmentation.
        frames = self.spatial_sampling(
            frames,
            spatial_idx=spatial_sample_index,
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
        )

        # Color Jittering
        frames = color_jitter(frames, 0.4, 0.4, 0.4)

        # Perform color normalization.
        frames = frames - torch.tensor(MEAN).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        frames = frames / torch.tensor(STD).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return frames

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self.valid_indices)

    def spatial_sampling(
        self,
        frames,
        spatial_idx=-1,
        min_scale=256,
        max_scale=320,
        crop_size=224,
    ):
        """
        Perform spatial sampling on the given video frames. If spatial_idx is
        -1, perform random scale, random crop, and random flip on the given
        frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
        with the given spatial_idx.
        Args:
            frames (tensor): frames of images sampled from the video. The
                dimension is `num frames` x `height` x `width` x `channel`.
            spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
                or 2, perform left, center, right crop if width is larger than
                height, and perform top, center, buttom crop if height is larger
                than width.
            min_scale (int): the minimal size of scaling.
            max_scale (int): the maximal size of scaling.
            crop_size (int): the size of height and width used to crop the
                frames.
        Returns:
            frames (tensor): spatially sampled frames.
        """
        assert spatial_idx in [-1, 0, 1, 2]
        if spatial_idx == -1:
            frames, _ = random_short_side_scale_jitter(
                frames, min_scale, max_scale
            )
            frames, _ = random_crop(frames, crop_size)
            frames, _ = horizontal_flip(0.5, frames)
        else:
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale, crop_size}) == 1
            frames, _ = random_short_side_scale_jitter(
                frames, min_scale, max_scale
            )
            frames, _ = uniform_crop(frames, crop_size, spatial_idx)
        return frames