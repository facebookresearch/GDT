#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import av
import torch as th
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import random
import ffmpeg
import time
import re
import csv
import pickle
from joblib import Parallel, delayed
import subprocess

def valid_video(root_dir, vid_idx, video_id):
    vid_path = os.path.join(root_dir, video_id + '.mp4')
    try:
        probe = ffmpeg.probe(vid_path)
        video_stream = next((
            stream for stream in probe['streams'] if stream['codec_type'] == 'video'), 
            None
        )
        if video_stream and float(video_stream['duration']) > 4.1:
            print(f"{vid_idx}: True", flush=True)
            return True
        else:
            print(f"{vid_idx}: False (duration short)", flush=True)
            return False
    except:
        print(f"{vid_idx}: False", flush=True)
        return False


def filter_videos(root_dir, vid_paths):
    all_indices = Parallel(n_jobs=30)(delayed(valid_video)(root_dir, vid_idx, vid_paths[vid_idx][0]) for vid_idx in range(0, len(vid_paths)))
    valid_indices = [i for i, val in enumerate(all_indices) if val]
    return valid_indices


class HT100M_Dataset(Dataset):
    """HowTo100M Video-Text loader."""

    def __init__(
        self,
        csv_file='data/howto.csv',
        video_root='/datasets01/HowTo100M/022520/videos',
        caption_root='/private/home/mandelapatrick/data/howto100m_csv',
        token_to_word_path='data/dict.npy',
        min_time=4.0,
        fps=16,
        num_frames=16,
        size=224,
        crop_only=False,
        center_crop=True,
        benchmark=False,
        max_words=20,
        num_candidates=1,
        random_left_right_flip=False,
        num_clips=2
    ):
        """
        Args:
        """
        print("Loading HT100M dataset")
        assert isinstance(size, int)

        # Get csv file
        csv_file = os.path.join(os.path.dirname(__file__), csv_file)
        if not os.path.exists(csv_file):
            i = 0
            file_list = []
            for file_name in os.listdir(video_root):
                if i % 1000 == 0:
                    print(i, file_name)
                file_list.append(file_name)
                i += 1
            
            with open(csv_file, 'w', newline='') as outcsv:
                fieldnames = ['video_id']
                writer = csv.DictWriter(outcsv, fieldnames=fieldnames)
                writer.writeheader()
                for id, vid_id in enumerate(file_list):
                    if i % 1000 == 0:
                        print(i, flush=True)
                    writer.writerow({'video_id': vid_id.split('.')[0]})
        
        # Get video paths
        with open(csv_file, newline='') as f:
            reader = csv.reader(f)
            self._path_to_videos = list(reader)

        # Get valid indices
        vid_valid_file = os.path.join(os.path.dirname(__file__), 'data/howto_valid_filtered_audio.pkl')
        if not os.path.exists(vid_valid_file):
            self.valid_indices = filter_videos(video_root, self._path_to_videos)
            with open(vid_valid_file, 'wb') as handle:
                pickle.dump(
                    self.valid_indices, 
                    handle, 
                    protocol=pickle.HIGHEST_PROTOCOL
                )
        else:
            with open(vid_valid_file, 'rb') as handle:
                self.valid_indices = pickle.load(handle)

        self.video_root = video_root
        self.caption_root = caption_root
        self.min_time = min_time
        self.size = size
        self.num_frames = num_frames
        self.fps = fps
        self.num_sec = self.num_frames / float(self.fps)
        self.crop_only = crop_only
        self.center_crop = center_crop
        self.benchmark = benchmark
        self.max_words = max_words
        token_to_word = np.load(os.path.join(os.path.dirname(__file__), token_to_word_path))
        self.word_to_token = {}
        for i, t in enumerate(token_to_word):
            self.word_to_token[t] = i + 1
        self.num_candidates = num_candidates
        self.random_flip = random_left_right_flip
        self.num_clips = num_clips
        self._num_retries = 10
        self.num_reverse_clips = 2

    def __len__(self):
        return len(self.valid_indices)

    def _get_video_ffmpeg(self, video_path, start, end):
        start_seek = random.randint(start, int(max(start, end - self.num_sec)))
        cmd = (
            ffmpeg
            .input(video_path, ss=start_seek, t=self.num_sec + 0.1)
            .filter('fps', fps=self.fps)
        )
        if self.center_crop:
            aw, ah = 0.5, 0.5
        else:
            aw, ah = random.uniform(0, 1), random.uniform(0, 1)
        if self.crop_only:
            cmd = (
                cmd.crop('(iw - {})*{}'.format(self.size, aw),
                         '(ih - {})*{}'.format(self.size, ah),
                         str(self.size), str(self.size))
            )
        else:
            cmd = (
                cmd.crop('(iw - min(iw,ih))*{}'.format(aw),
                         '(ih - min(iw,ih))*{}'.format(ah),
                         'min(iw,ih)',
                         'min(iw,ih)')
                .filter('scale', self.size, self.size)
            )
        if self.random_flip and random.uniform(0, 1) > 0.5:
            cmd = cmd.hflip()
        out, _ = (
            cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run(capture_stdout=True, quiet=True)
        )
        video = np.frombuffer(out, np.uint8).reshape([-1, self.size, self.size, 3])
        video = th.from_numpy(video)
        video = video.permute(3, 0, 1, 2)
        if video.shape[1] < self.num_frames:
            zeros = th.zeros((3, self.num_frames - video.shape[1], self.size, self.size), dtype=th.uint8)
            video = th.cat((video, zeros), axis=1)
        # return video[:, :self.num_frames]
        video = video.float()
        video = video / 255.0
        return video[:, :self.num_frames], start_seek

    def _split_text(self, sentence):
        w = re.findall(r"[\w']+", str(sentence))
        return w

    def _words_to_token(self, words):
        words = [self.word_to_token[word] for word in words if word in self.word_to_token]
        if words:
            we = self._zero_pad_tensor_token(th.LongTensor(words), self.max_words)
            return we
        else:
            return th.zeros(self.max_words, dtype=th.long)

    def _zero_pad_tensor_token(self, tensor, size):
        if len(tensor) >= size:
            return tensor[:size]
        else:
            zero = th.zeros(size - len(tensor)).long()
            return th.cat((tensor, zero), dim=0)

    def words_to_ids(self, x):
        return self._words_to_token(self._split_text(x))

    def _find_nearest_candidates(self, caption, ind):
        start, end = ind, ind
        diff = caption['end'][end] - caption['start'][start]
        n_candidate = 1
        while n_candidate < self.num_candidates:
           if start == 0:
               return 0
           elif end == len(caption) - 1:
               return start - (self.num_candidates - n_candidate)
           elif caption['end'][end] - caption['start'][start - 1] < caption['end'][end + 1] - caption['start'][start]:
               start -= 1
           else:
               end += 1
           n_candidate += 1
        return start

    def _get_text(self, caption):
        cap = pd.read_csv(caption)
        ind = random.randint(0, len(cap) - 1)
        if self.num_candidates == 1:
            words = self.words_to_ids(cap['text'].values[ind])
        else:
            words = th.zeros(self.num_candidates, self.max_words, dtype=th.long)
            cap_start = self._find_nearest_candidates(cap, ind)
            for i in range(self.num_candidates):
                words[i] = self.words_to_ids(cap['text'].values[max(0, min(len(cap['text']) - 1, cap_start + i))])
        start, end = cap['start'].values[ind], cap['end'].values[ind]
        #TODO: May need to be improved for edge cases. 
        if end - start < self.min_time:
            diff = self.min_time - end + start
            start = max(0, start - diff / 2)
            end = start + self.min_time 
        return words, int(start), int(end) 

    def __getitem__(self, idx):
        
        for i_try in range(self._num_retries):
            
            # Get video id and path
            index_capped = self.valid_indices[idx]
            video_id = self._path_to_videos[index_capped][0]
            video_path = os.path.join(self.video_root, video_id + '.mp4')
            video_list = []
            text_list = []
            audio_list = []
    
            while len(video_list) < self.num_clips:
                # Get caption
                text, start, end = self._get_text(os.path.join(self.caption_root, video_id + '.csv'))

                # Decode video
                video = None
                try:
                    video, start_sec = self._get_video_ffmpeg(video_path, start, end)
                except Exception as e:
                    print(f"Failed to load video from {video_path} with error {e}")
                if video is None:
                    # let's try another video
                    if i_try > self._num_retries // 2:
                        idx = random.randint(0, len(self.valid_indices) - 1)
                    break
                
                video_list.append(video)
                text_list.append(text)

            if len(video_list) == self.num_clips:
                break

        if i_try == self._num_retries - 1:
            raise RuntimeError(
                "Failed to fetch video after {} retries.".format(
                    self._num_retries
                )
            )

        # Add reversal option
        for i in range(self.num_clips):
            # Clone frames and spec
            frames = video_list[i].clone()
            text = text_list[i].clone()

            for r_ix in range(self.num_reverse_clips):
                # Reverse audio and video
                if r_ix % 2 == 1:
                    frames = frames.flip(1) # C T H W 
                    text = text.flip(0) # T
                        
                    video_list.append(frames)
                    text_list.append(text)
        
        if self.num_reverse_clips == 2:
            video_list = [video_list[i] for i in [0, 2, 1, 3]]
            text_list = [text_list[i] for i in [0, 2, 1, 3]]

        if self.num_clips > 1:
            video = th.cat(video_list, dim=0)
            text = th.cat(text_list, dim=0)
        else:
            video = video_list[0]
            text = text_list[0]

        label = 0
        vid_idx = index_capped	

        return video, text, label, vid_idx, index_capped