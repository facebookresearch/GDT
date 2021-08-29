#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import av
import ffmpeg
from joblib import Parallel, delayed
from multiprocessing import Manager
import numpy as np
import os
import pickle
import torch
import torch.utils.data
import glob

try: 
    from decoder import decode
except:
    from .decoder import decode

try: 
    from video_transforms import clip_augmentation
except:
    from .video_transforms import clip_augmentation

# Enable multi thread decoding.
ENABLE_MULTI_THREAD_DECODE = True

# Decoding backend, options include `pyav` or `torchvision`
DECODING_BACKEND = 'pyav'

ROOT_DIR = {
    'kinetics': '/datasets01/kinetics/070618/',
    'kinetics600': '/datasets01/kinetics/070618/600/',
    'ucf101': '/private/home/mandelapatrick/data/ucf101',
    'hmdb51': '/datasets01/hmdb51/112018/data/'
}

MODE_DIR = {
    'kinetics': {
        'train': 'train_avi-480p',
        'val': 'val_avi-480p',
        'test': 'val_avi-480p'
    },
    'kinetics600': {
        'train': 'train',
        'val': 'val'
    },
    'ucf101': {
        'train': '',
        'test': ''
    },
    'hmdb51': {
        'train': '',
        'test': ''
    },
}


def select_fold_hmdb51(video_list, annotation_path, fold, train, num_shots=-1):
        print(f"Getting HMDB51 dataset. Train Mode: {train}, fold: {fold}", flush=True)
        target_tag = 1 if train else 2
        name = "*test_split{}.txt".format(fold)
        files = glob.glob(os.path.join(annotation_path, name))
        selected_files = []
        for f in files:
            with open(f, "r") as fid:
                data = fid.readlines()
                data = [x.strip().split(" ") for x in data]
                data = [x[0] for x in data if int(x[1]) == target_tag]
                selected_files.extend(data)
        selected_files = set(selected_files)
        indices = [i for i in range(len(video_list)) if os.path.basename(video_list[i]) in selected_files]
        return indices


def select_fold_ucf101(root, video_list, annotation_path, fold, train, num_shots=-1):
        name = "train" if train else "test"
        name = "{}list{:02d}.txt".format(name, fold)
        f = os.path.join(annotation_path, name)
        print(f)
        selected_files = []
        with open(f, "r") as fid:
            data = fid.readlines()
            data = [x.strip().split(" ") for x in data]
            data = [x[0] for x in data]
            selected_files.extend(data)
        selected_files = set(selected_files)
        indices = [i for i in range(len(video_list)) if video_list[i][len(root):] in selected_files]
        return indices


def valid_video(vid_idx, vid_path, decode_audio, num_sec):
    try:
        probe = ffmpeg.probe(vid_path)
        video_stream = next((
            stream for stream in probe['streams'] if stream['codec_type'] == 'video'), 
            None
        )
        if decode_audio:
            audio_stream = next((
                stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), 
                None
            )
            if audio_stream and video_stream and float(video_stream['duration']) > num_sec and float(audio_stream['duration']) > num_sec:
                print(f"{vid_idx}: True", flush=True)
                return True
            else:
                print(f"{vid_idx}: False (duration short/ no audio)", flush=True)
                return False
        else:
            if video_stream and float(video_stream['duration']) > num_sec:
                print(f"{vid_idx}: True", flush=True)
                return True
            else:
                print(f"{vid_idx}: False (duration short/ no audio)", flush=True)
                return False
    except:
        print(f"{vid_idx}: False", flush=True)
        return False


def filter_videos(vid_paths, decode_audio=True, num_sec=1.1):
    all_indices = Parallel(n_jobs=30)(delayed(valid_video)(vid_idx, vid_paths[vid_idx], decode_audio, num_sec) for vid_idx in range(len(vid_paths)))        
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


class AVideoDataset(torch.utils.data.Dataset):
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
        root_dir=None,
        mode='train',
        decode_audio=False,
        num_train_clips=1,
        # settings for downstream tasks
        num_ensemble_views=10,
        num_spatial_crops=3,
        center_crop=False,
        fold=1,
        ucf101_annotation_path='/datasets01/ucf101/112018/ucfTrainTestlist/',
        hmdb51_annotation_path='/datasets01/hmdb51/112018/splits/',
        args=None,
    ):
        self.ds_name = ds_name
        self.mode = mode
        self.path_to_data_dir='datasets/data' # local directory to store pickle of valid videos
        self.num_frames=args.num_frames   # number of frames
        self.target_fps=args.target_fps   # frames per second
        self.sample_rate=args.sample_rate # audio sample rate
        self.num_train_clips=num_train_clips # num of clips to sample per video during training
        #* video related params
        self.train_crop_size=args.train_crop_size
        self.test_crop_size=args.test_crop_size
        ##* video croppings
        self.multi_crop=args.multi_crop # whether to get multiple crops per video
        self.use_random_resize_crop=args.use_random_resize_crop # if not, it just does random cropping.
        ##* downstream task cropping settings
        self.center_crop = center_crop # only center crop?
        self.num_ensemble_views = num_ensemble_views # 10 crops in time
        self.num_spatial_crops = num_spatial_crops # 3 in space
        ##* video augmentations

        self.colorjitter=args.colorjitter
        self.use_grayscale=args.use_grayscale
        self.use_gaussian=args.use_gaussian

        #* audio settings
        self.decode_audio=decode_audio
        self.num_sec=args.num_sec_aud
        self.aud_sample_rate=args.aud_sample_rate
        self.aud_spec_type=args.aud_spec_type
        self.use_volume_jittering=args.use_volume_jittering
        self.use_temporal_jittering=args.use_audio_temp_jittering
        self.z_normalize=args.z_normalize

        #* other
        self.num_data_samples=None # if we want to artificially limit size of dataset

        # ucf101 and hmdb51
        self.ucf101_annotation_path = ucf101_annotation_path
        self.hmdb51_annotation_path = hmdb51_annotation_path

        assert mode in ["train","val","test"], "Split '{}' not supported for '{}'".format(mode, ds_name)
        if self.train_crop_size in [112, 128]:
            train_jitter_scales = (128, 160)
        else:
            train_jitter_scales = (256, 320)
        self.train_jitter_scales = train_jitter_scales
        if root_dir is None:
            self.data_prefix = os.path.join(ROOT_DIR[ds_name], MODE_DIR[ds_name][mode])
        else:
            self.data_prefix = root_dir
        self._video_meta = {}
        self.fold = fold # ucf101 and hmdb51

        # Get classes
        classes = list(sorted(glob.glob(os.path.join(self.data_prefix, '*'))))
        classes = [os.path.basename(i) for i in classes]
        self.class_to_idx = {classes[i]: i for i in range(len(classes))}

        # For training or validation mode, one single clip is sampled from every video.
        # For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every video.
        # For every clip, NUM_SPATIAL_CROPS is cropped spatially from the frames.
        if self.mode in ["train", "val"]:
            self._num_clips = self.num_train_clips
        elif self.mode in ["test"]:
            self._num_clips = (
                    self.num_ensemble_views * self.num_spatial_crops
            )

        self.manager = Manager()
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
        if not os.path.exists(path_to_file):
            files = list(sorted(glob.glob(os.path.join(self.data_prefix, '*', '*'))))
            with open(path_to_file, 'w') as f:
                for item in files:
                    f.write("%s\n" % item)

        # Get list of indices and labels
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
        if self.ds_name in ['kinetics', 'kinetics600']:
            if self.mode == 'train':
                vid_valid_file = f'{self.path_to_data_dir}/{self.ds_name}_valid.pkl'
                if os.path.exists(vid_valid_file):
                    with open(vid_valid_file, 'rb') as handle:
                        self.valid_indices = pickle.load(handle)
                else:
                    self.valid_indices = filter_videos(self._path_to_videos, decode_audio=self.decode_audio)
                    with open(vid_valid_file, 'wb') as handle:
                        pickle.dump(
                            self.valid_indices,
                            handle,
                            protocol=pickle.HIGHEST_PROTOCOL
                        )
                if self.num_data_samples is not None:
                    rand_indices = np.random.choice(range(len(self.valid_indices)), self.num_data_samples, replace=False)
                    self.valid_indices = np.array(self.valid_indices)[rand_indices]
            else:
                # self.valid_indices = [i for i in range(0, len(self._path_to_videos))]
                vid_valid_file = f'{self.path_to_data_dir}/{self.ds_name}_valid_{self.mode}_{self.decode_audio}.pkl'
                if os.path.exists(vid_valid_file):
                    with open(vid_valid_file, 'rb') as handle:
                        self.valid_indices = pickle.load(handle)
                else:
                    self.valid_indices = filter_videos(self._path_to_videos, decode_audio=self.decode_audio)
                    with open(vid_valid_file, 'wb') as handle:
                        pickle.dump(
                            self.valid_indices,
                            handle,
                            protocol=pickle.HIGHEST_PROTOCOL
                        )
            print(f"Total number of videos: {len(self._path_to_videos)}, Valid videos: {len(self.valid_indices)}", flush=True)
        else: # ucf101 and hmdb-51
            if self.ds_name == 'ucf101':
                train = True if self.mode == 'train' else False
                self.valid_indices = select_fold_ucf101(self.data_prefix, self._path_to_videos, self.ucf101_annotation_path, self.fold, train)
            elif self.ds_name == 'hmdb51':
                train = True if self.mode == 'train' else False
                self.valid_indices = select_fold_hmdb51(self._path_to_videos, self.hmdb51_annotation_path, self.fold, train)
            else:
                assert(False)
            print(f"Total number of videos: {len(self._path_to_videos)}, Valid videos: {len(self.valid_indices)}", flush=True)


        # Make lists a Manager objects
        self._path_to_videos = self.manager.list(self._path_to_videos)
        self.valid_indices = self.manager.list(self.valid_indices)


    def __getitem__(self, index):
        index_capped = index
        index = self.valid_indices[index_capped]
        if self.mode in ["train", "val"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
            spatial_sample_index = -1
            max_scale = self.train_jitter_scales[1]
            if self.center_crop:
                spatial_sample_index = 1
                max_scale = self.train_crop_size
        elif self.mode in ["test"]:
            temporal_sample_index = (
                    self._spatial_temporal_idx[index] // self.num_spatial_crops
            )
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = (
                    self._spatial_temporal_idx[index] % self.num_spatial_crops
            )
            min_scale, max_scale, crop_size = [self.test_crop_size] * 3
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            # assert len({min_scale, max_scale, crop_size}) == 1
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )

        # Get number of clips
        num_clips = 1
        V = []
        A = []

        for i in range(num_clips):
            # Try to decode and sample a clip from a video.
            video_container = get_video_container(
                self._path_to_videos[index],
                ENABLE_MULTI_THREAD_DECODE,
                DECODING_BACKEND,
            )

            # Decode video. Meta info is used to perform selective decoding.
            frames, audio = decode(
                self._path_to_videos[index],
                video_container,
                self.sample_rate,
                self.num_frames,
                temporal_sample_index,
                self.num_ensemble_views,
                video_meta=self._video_meta[index],
                target_fps=int(self.target_fps),
                backend=DECODING_BACKEND,
                max_spatial_scale=max_scale,
                decode_audio=self.decode_audio,
                aud_sample_rate=self.aud_sample_rate,
                aud_spec_type=self.aud_spec_type,
                use_volume_jittering=self.use_volume_jittering,
                num_sec=int(self.num_sec),
                use_temporal_jittering=self.use_temporal_jittering,
                z_normalize=self.z_normalize,
            )

            # Perform data augmentation on video clip.
            if self.multi_crop:
                multi_crop_clips = 2
            else:
                multi_crop_clips = 1
            use_random_resize_crop = (self.use_random_resize_crop) and (self.mode == 'train')
            for j in range(multi_crop_clips):
                min_scale, max_scale, crop_size = (self.train_jitter_scales[0], 
                    self.train_jitter_scales[1], self.train_crop_size)
                if use_random_resize_crop:
                    min_scale, max_scale, crop_size = 0.14, 1.0, self.train_crop_size
                V.append(clip_augmentation(
                    frames.clone(),
                    spatial_idx=spatial_sample_index,
                    min_scale=min_scale,
                    max_scale=max_scale,
                    crop_size=crop_size,
                    colorjitter=self.colorjitter,
                    use_grayscale=self.use_grayscale,
                    use_gaussian=self.use_gaussian,
                    use_random_resize_crop=use_random_resize_crop
                ))
            A.append(audio)

        # Get labels and indices
        label = self._labels[index]
        vid_idx = self._vid_indices[index]

        # return results
        if not self.multi_crop:
            V = torch.cat(V, dim=0)
        if self.decode_audio:
            audio = torch.cat(A, dim=0)
            return V, audio, label, index_capped, vid_idx
        else:
            return V, label, index_capped, vid_idx


    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self.valid_indices)
