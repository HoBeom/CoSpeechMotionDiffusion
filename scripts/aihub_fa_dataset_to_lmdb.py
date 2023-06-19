import argparse
import glob
import os
import os.path as osp
from pathlib import Path

import json
import math
import numpy as np
import re
import librosa
import lmdb
import pyarrow
from sklearn.pipeline import Pipeline
import joblib as jl
from collections import defaultdict

from pymo.preprocessing import *
from pymo.parsers import BVHParser

try:
    from rich.progress import track as tqdm
except ImportError:
    from tqdm import tqdm

class ForcedAlignmentDataWrapper:

    def __init__(self, pkl_path, merge_sentence=True):
        print(f'load {pkl_path}')
        self.pkl_file = jl.load(pkl_path)
        self.video_list = self.pkl_file.keys()
        self.video_list = sorted(list(self.video_list))
        self.iter_idx = -1
        self.clip_dict = defaultdict(list)
        if merge_sentence:
            self.merge_sentence()
        else:
            self.make_clip_list()
        self.vaild_video_check()
        self.clip_len = 0
        self.clip_per_sentence = defaultdict(int)
        for video_name in self.video_list:
            self.clip_len += len(self.clip_dict[video_name])
            self.clip_per_sentence[len(self.clip_dict[video_name])] += 1
        print(f'video_num: {len(self.video_list)}')
        print(f'clip_len: {self.clip_len}')
        print(f'clip_per_sentence: {sorted(self.clip_per_sentence.items(), key=lambda x: x[0])}')

    def __len__(self):
        return len(self.video_list)
    
    def __iter__(self):
        return self

    def __next__(self):
        self.iter_idx += 1
        if self.iter_idx >= len(self.video_list):
            raise StopIteration
        video_name = self.video_list[self.iter_idx]
        return video_name, self.clip_dict[video_name]

    def vaild_video_check(self):
        for video_name in self.video_list:
            if len(self.clip_dict[video_name]) == 0:
                print(f'invalid video: {video_name}')
                raise ValueError

    def make_clip_list(self):
        for video_name in self.video_list:
            for sentence_idx, data in self.pkl_file[video_name].items():
                self.clip_dict[video_name].append((sentence_idx, data))

    def merge_sentence(self):
        for video_name in self.video_list:
            pre_idx = -1
            clip = []
            for sentence_idx, data in self.pkl_file[video_name].items():
                if sentence_idx == pre_idx + 1:
                    clip.append((sentence_idx, data))
                    pre_idx = sentence_idx
                else:
                    if pre_idx != -1:
                        self.clip_dict[video_name].append(clip)
                    clip = []
                    clip.append((sentence_idx, data))
                    pre_idx = sentence_idx
            if len(clip) != 0:
                self.clip_dict[video_name].append(clip)




class JsonSubtitleWrapper:

    def __init__(self, subtitle_path):
        self.subtitle = []
        self.load_json_subtitle(subtitle_path)

    def get(self):
        return self.subtitle
    
    def get_duration(self):
        return self.json_file["Info"]["Audio_Info"]["audio_duration"]
    
    def get_total_video_frames(self):
        return len(self.get_keypoints_3d_face())

    def get_keypoints_3d(self):
        return self.json_file["Motion"]["Keypoints"]["keypoints_3d"][0]["body"]
    
    def get_keypoints_3d_face(self):
        return self.json_file["Motion"]["Keypoints"]["keypoints_3d"][0]["face"]


    def load_json_subtitle(self, subtitle_path):
        try:
            with open(subtitle_path, 'r', encoding='cp949') as file:
                self.json_file = json.load(file)
                for sentence in self.json_file['Transcript']['Sentences']:
                    self.subtitle.append(sentence)
                    #  {'emotion': 'neutral',
                    #  'end_time': 10.11,
                    #  'sentence_text': '오늘 시험공부든 요즘 시험공부 땜에 스트레스받아요.',
                    #  'speaker_ID': 'S161',
                    #  'start_time': 6.95}
        except FileNotFoundError:
            self.subtitle = None

def normalize_string_ko(s):
    """ lowercase, trim, and remove non-letter characters """
    s = s.strip()
    # s = re.sub(r"([,.!?])", r" \1 ", s)  # isolate some marks
    # s = re.sub(r"[^ㄱ-ㅎㅏ-ㅣ가-힣,.!?]+", r" ", s)  # replace other characters with whitespace
    s = re.sub(r"\s+", r" ", s).strip()
    return s

class AudioWrapper:
    def __init__(self, filepath):
        self.y, self.sr = librosa.load(filepath, mono=True, sr=16000, res_type='kaiser_fast')
        self.n = len(self.y)
        print('librosa', self.n)

    def extract_audio_feat(self, video_total_frames, video_start_frame, video_end_frame):
        # roi
        start_frame = math.floor(video_start_frame / video_total_frames * self.n)
        end_frame = math.ceil(video_end_frame / video_total_frames * self.n)
        y_roi = self.y[start_frame:end_frame]

        # feature extraction
        melspec = librosa.feature.melspectrogram(
            y=y_roi, sr=self.sr, n_fft=1024, hop_length=512, power=2)
        log_melspec = librosa.power_to_db(melspec, ref=np.max)  # mels x time

        log_melspec = log_melspec.astype('float16')
        y_roi = y_roi.astype('float16')

        # DEBUG
        # print('spectrogram shape: ', log_melspec.shape)

        return log_melspec, y_roi

def make_lmdb_gesture_dataset(base_path):
    out_path = osp.join('data', 'aihub', 'lmdb')
    if not osp.exists(out_path):
        os.makedirs(out_path)

    map_size = 1024 * 100  # in MB
    map_size <<= 20  # in B
    print(f'map_size: {map_size}')
    db = [
        lmdb.open(osp.join(out_path, 'lmdb_train'), map_size=map_size),
        lmdb.open(osp.join(out_path, 'lmdb_test'), map_size=map_size),
    ]

    fa = [
        ForcedAlignmentDataWrapper(osp.join(base_path, 'train_fa_data.pkl')),
        ForcedAlignmentDataWrapper(osp.join(base_path, 'test_fa_data.pkl')),
    ]
    audio_split = [
        osp.join(base_path, 'wav', 'train'),
        osp.join(base_path, 'wav', 'val'),
    ]

    anno_split = [
        osp.join(base_path, 'json', 'train'),
        osp.join(base_path, 'json', 'val'),
    ]
    # delete existing files
    for i in range(2):
        with db[i].begin(write=True) as txn:
            txn.drop(db[i].open_db())

    # all_poses = []
    # all_dir_vecs = []
    # bvh_files = sorted(glob.glob(gesture_path + "/*.bvh"))
    save_idx = 0
    for mode_idx, paths in enumerate(zip(db, fa, audio_split, anno_split)):
        split_db, fa_wrapper, audio_path, anno_path = paths
        for name, segments in tqdm(fa_wrapper): 
            # segment contains multiple sentences

            # load json subtitles
            json_path = osp.join(anno_path, name + '.json')
            if osp.isfile(json_path):
                json_wrapper = JsonSubtitleWrapper(json_path)
                subtitle = json_wrapper.get()
                poses = np.array(json_wrapper.get_keypoints_3d())
                # poses = poses[:, joint_label_index, :] # 10 x 3
                video_duration = json_wrapper.get_duration()
                video_total_frames = json_wrapper.get_total_video_frames()
                print(f"video_duration: {video_duration}, video_total_frames: {video_total_frames}")
            else:
                continue

            # load audio
            wav_path = osp.join(audio_path, '{}.wav'.format(name))
            if osp.isfile(wav_path):
                # audio_raw, audio_sr = librosa.load(wav_path, mono=True, sr=16000, res_type='kaiser_fast')
                # mel audio feature
                audio_wrapper = AudioWrapper(wav_path)
                # audio_feat, audio_raw = audio_wrapper.extract_audio_feat(len(poses), 0, len(poses))
            else:
                continue

            clips = {'vid': name, 'clips': []}
            for segment_idx, segment in enumerate(segments):

                # {'words': word_list,
                # #  'poses': poses.tolist(),
                # 'skeletons_3d': poses,
                # 'audio_raw': audio_raw,
                # 'audio_feat': audio_feat,
                # 'start_frame_no': 0,
                # 'end_frame_no': len(poses) - 1,
                # 'start_time': 0,
                # 'end_time': json_wrapper.get_duration()
                # })
                # print(segment)
                # [(0, ['0.0 8.68 <SIL>\n', '8.68 9.2 요즘\n', '9.2 9.81 시험공부\n', '9.81 10.21 때문에\n', '10.21 11.14 스트레스받아요\n', '11.14 11.39 <SIL>\n']), 
                # (1, ['0.0 0.3 <SIL>\n', '0.3 0.82 관심도\n', '0.82 1.23 없는\n', '1.23 1.62 체육\n', '1.62 1.93 과학\n', '1.93 2.3 과목\n', '2.3 2.82 공부도\n', 
                # '2.82 3.04 해야\n', '3.04 3.38 되고\n', '3.38 3.74 <SIL>\n', '3.74 4.01 하기\n', '4.01 4.32 싫은\n', '4.32 4.83 
                # 공부도\n', '4.83 5.24 억지로\n', '5.24 5.43 해야\n', '5.43 5.67 되는\n', '5.67 5.77 게\n', '5.77 5.99 너무\n', '5.99 6.84 고민이에요\n', '6.84 7.08 <SIL>\n'])]
                # print(subtitle[segment[0][0]])
                # {'emotion': 'angry', 'end_time': 11.39, 'sentence_text': '요즘 시험공부 때문에 스트레스받아요.', 'speaker_ID': 'S160', 'start_time': 8.45}

                start_time = subtitle[segment[0][0]]['start_time']
                end_time = subtitle[segment[-1][0]]['end_time']
                # print(start_time, end_time)
                # print(subtitle[segment[0][0]]['end_time'])
                if end_time - start_time < 3: 
                    # https://github.com/youngwoo-yoon/youtube-gesture-dataset/blob/09e2c5fc0c51d048ce03eaf1ba969c0ee983e11d/script/clip_filter.py#L101
                    print('too short')
                    continue
                
                start_frame = math.floor(start_time / video_duration * video_total_frames)
                end_frame = math.ceil(end_time / video_duration * video_total_frames)
                # save subtitles and skeletons
                clip_pose = poses[start_frame:end_frame]
                if np.isnan(clip_pose).any():
                    print('nan pose')
                    # raise ValueError
                    continue

                # sentence preprocessing
                word_list = []
                for si, fa_data in segment:
                    # fa 과정에서 wav 파일 스플릿 버그로 인해 첫번째 문장의 시작 시간이 0으로 나옴
                    if si == 0:
                        sen_s = start_time
                    else:
                        sen_s = float(subtitle[si]['start_time'])
                    # sen_e = float(subtitle[si]['end_time'])
                    for fi, data in enumerate(fa_data):
                        st, en, word = data.strip().split(' ')
                        # if word == '<SIL>':
                        #     continue
                        if si == 0:
                            if fi == 0: # '<SIL>'
                                token_s = sen_s
                                token_e = round(float(en), 4)
                            else:
                                token_s = round(float(st), 4)
                                token_e = round(float(en), 4)
                        else:
                            token_s = round(float(st) + sen_s, 4)
                            token_e = round(float(en) + sen_s, 4)
                        word_list.append([word, token_s, token_e])
                    # 문장과 문장 사이에 <SIL>을 삽입?
                    if si == 0:
                        print(word_list, start_time, end_time)
                # print(word_list)
                # [['<SIL>', 8.45, 17.13], ['요즘', 17.13, 17.65], ['시험공부', 17.65, 18.26], 
                # ['때문에', 18.26, 18.66], ['스트레스받아요', 18.66, 19.59], ['<SIL>', 19.59, 19.84], 
                # ['<SIL>', 11.61, 11.91], ['관심도', 11.91, 12.43], ['없는', 12.43, 12.84], ['체육', 12.84, 13.23], ['과학', 13.23, 13.54], ['과목', 13.54, 13.91], ['공부도', 13.91, 14.43], ['해야', 14.43, 14.65], ['되고', 14.65, 14.99], ['<SIL>', 14.99, 15.35], 
                # ['하기', 15.35, 15.62], ['싫은', 15.62, 15.93], ['공부도', 15.93, 16.44], ['억지로', 16.44, 16.85], ['해야', 16.85, 17.04], ['되는', 17.04, 17.28], ['게', 17.28, 17.38], ['너무', 17.38, 17.6], ['고민이에요', 17.6, 18.45], ['<SIL>', 18.45, 18.69]]
                audio_feat, audio_raw = audio_wrapper.extract_audio_feat(video_duration, start_time, end_time)

                audio_raw = np.asarray(audio_raw, dtype=np.float16)
                audio_feat = np.asarray(audio_feat, dtype=np.float16)

                clips['clips'].append(
                    {'words': word_list,
                    #  'poses': poses.tolist(),
                    'skeletons_3d': clip_pose.astype('float16'),
                    'audio_raw': audio_raw,
                    'audio_feat': audio_feat,
                    'start_frame_no': start_frame,
                    'end_frame_no': end_frame,
                    'start_time': start_time,
                    'end_time': end_time,
                    'segment_idx': segment_idx,
                    })

                # clip pose nan check
                save_idx += 1


            print(f'len(clips): {len(clips["clips"])}')
            # write to db
            with split_db.begin(write=True) as txn:
                if len(clips['clips']) > 0:
                    k = '{:010}'.format(save_idx).encode('ascii')
                    v = pyarrow.serialize(clips).to_buffer()
                    txn.put(k, v)

    # close db
    for i in range(2):
        db[i].sync()
        db[i].close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_path", default='data/aihub', type=Path)
    args = parser.parse_args()

    make_lmdb_gesture_dataset(args.db_path)
