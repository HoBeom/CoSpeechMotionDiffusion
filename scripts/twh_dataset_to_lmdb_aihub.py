import argparse
import glob
import os
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

from pymo.preprocessing import *
from pymo.parsers import BVHParser

try:
    from rich.progress import track as tqdm
except ImportError:
    from tqdm import tqdm

# 18 joints (only upper body)
# target_joints = ['b_spine0', 'b_spine1', 'b_spine2', 'b_spine3', 
# 'b_l_shoulder', 'b_l_arm', 'b_l_arm_twist', 'b_l_forearm', 
# 'b_l_wrist_twist', 'b_l_wrist', 'b_r_shoulder', 'b_r_arm', 
# 'b_r_arm_twist', 'b_r_forearm', 'b_r_wrist_twist', 'b_r_wrist', 'b_neck0', 'b_head']

# 10 joints (only upper body)
target_joints = [
    "Hips",
    "Neck",
    "Head",
    "Head_Nub",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightArm",
    "RightForeArm",
    "RightHand",
]

# joint_label = ['Hip',
#  'Ab',
#  'Chest',
#  'Neck',
#  'Head',
#  'LShoulder',
#  'LUArm',
#  'LFArm',
#  'LHand',
#  'RShoulder',
#  'RUArm',
#  'RFArm',
#  'RHand',
#  'LThigh',
#  'LShin',
#  'LFoot',
#  'RThigh',
#  'RShin',
#  'RFoot',
#  'LToe',
#  'RToe']
joint_label_index = [0,2,3,4, 10,11,12, 6,7,8]


# 24 joints (upper and lower body excluding fingers)
# target_joints = ['body_world', 'b_root', 'b_l_upleg', 'b_l_leg', 'b_r_upleg', 'b_r_leg', 'b_spine0', 'b_spine1', 'b_spine2', 'b_spine3', 'b_l_shoulder', 'b_l_arm', 'b_l_arm_twist', 'b_l_forearm', 'b_l_wrist_twist', 'b_l_wrist', 'b_r_shoulder', 'b_r_arm', 'b_r_arm_twist', 'b_r_forearm', 'b_r_wrist_twist', 'b_r_wrist', 'b_neck0', 'b_head']

# 56 joints (upper and lower body including fingers)
# target_joints = ['body_world', 'b_root', 'b_l_upleg', 'b_l_leg', 'b_r_upleg', 'b_r_leg', 'b_spine0', 'b_spine1', 'b_spine2', 'b_spine3', 'b_l_shoulder', 'b_l_arm', 'b_l_arm_twist', 'b_l_forearm', 'b_l_wrist_twist', 'b_l_wrist', 'b_l_pinky1', 'b_l_pinky2', 'b_l_pinky3', 'b_l_ring1', 'b_l_ring2', 'b_l_ring3', 'b_l_middle1', 'b_l_middle2', 'b_l_middle3', 'b_l_index1', 'b_l_index2', 'b_l_index3', 'b_l_thumb0', 'b_l_thumb1', 'b_l_thumb2', 'b_l_thumb3', 'b_r_shoulder', 'b_r_arm', 'b_r_arm_twist', 'b_r_forearm', 'b_r_wrist_twist', 'b_r_wrist', 'b_r_thumb0', 'b_r_thumb1', 'b_r_thumb2', 'b_r_thumb3', 'b_r_pinky1', 'b_r_pinky2', 'b_r_pinky3', 'b_r_middle1', 'b_r_middle2', 'b_r_middle3', 'b_r_ring1', 'b_r_ring2', 'b_r_ring3', 'b_r_index1', 'b_r_index2', 'b_r_index3', 'b_neck0', 'b_head']

class JsonSubtitleWrapper:

    def __init__(self, subtitle_path):
        self.subtitle = []
        self.load_json_subtitle(subtitle_path)

    def get(self):
        return self.subtitle
    
    def get_duration(self):
        return self.json_file["Info"]["Audio_Info"]["audio_duration"]
    
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

def process_bvh(gesture_filename, tgt_fps=15, dump_pipeline=False):
    p = BVHParser()

    data_all = list()
    data_all.append(p.parse(gesture_filename))

    data_pipe = Pipeline([
        ('dwnsampl', DownSampler(tgt_fps=tgt_fps, keep_all=False)),
        # ('root', RootNormalizer()),
        ('jtsel', JointSelector(target_joints, include_root=False)),
        # ('mir', Mirror(axis='X', append=True)),
        ('np', Numpyfier())
    ])

    out_data = data_pipe.fit_transform(data_all)
    if dump_pipeline:
        jl.dump(data_pipe, os.path.join('./resource', 'data_pipe.sav'))

    # euler -> rotation matrix
    out_data = out_data.reshape((out_data.shape[0], out_data.shape[1], -1, 6))  # 3 pos (XYZ), 3 rot (ZXY)
    out_matrix = np.zeros((out_data.shape[0], out_data.shape[1], out_data.shape[2], 12))  # 3 pos, 1 rot matrix (9 elements)
    for i in range(out_data.shape[0]):  # mirror
        for j in range(out_data.shape[1]):  # frames
            for k in range(out_data.shape[2]):  # joints
                out_matrix[i, j, k, :3] = out_data[i, j, k, :3]  # positions
                r = R.from_euler('ZXY', out_data[i, j, k, 3:], degrees=True)
                out_matrix[i, j, k, 3:] = r.as_matrix().flatten()  # rotations
    out_matrix = out_matrix.reshape((out_data.shape[0], out_data.shape[1], -1))

    return out_matrix[0]

class AudioWrapper:
    def __init__(self, filepath):
        self.y, self.sr = librosa.load(filepath, mono=True, sr=16000, res_type='kaiser_fast')
        self.n = len(self.y)

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
    gesture_path = os.path.join(base_path, 'bvh')
    audio_path = os.path.join(base_path, 'wav')
    text_path = os.path.join(base_path, 'json')
    out_path = os.path.join(base_path, 'lmdb')
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    map_size = 1024 * 20  # in MB
    map_size <<= 30  # in B
    db = [lmdb.open(os.path.join(out_path, 'lmdb_train'), map_size=map_size),
          lmdb.open(os.path.join(out_path, 'lmdb_test'), map_size=map_size)]

    # delete existing files
    for i in range(2):
        with db[i].begin(write=True) as txn:
            txn.drop(db[i].open_db())

    all_poses = []
    bvh_files = sorted(glob.glob(gesture_path + "/*.bvh"))
    save_idx = 0
    for bvh_file in tqdm(bvh_files):
        name = os.path.split(bvh_file)[1][:-4]
        # print(name)

        # load skeletons
        # dump_pipeline = (save_idx == 2)  # trn_2022_v1_002 has a good rest finger pose
        # poses, poses_mirror = process_bvh(bvh_file)
        # poses = process_bvh(bvh_file)

        # load subtitles
        # tsv_path = os.path.join(text_path, name + '.tsv')
        # if os.path.isfile(tsv_path):
        #     subtitle = SubtitleWrapper(tsv_path).get()
        # else:
        #     continue

        # load json subtitles
        json_path = os.path.join(text_path, name + '.json')
        if os.path.isfile(json_path):
            json_wrapper = JsonSubtitleWrapper(json_path)
            subtitle = json_wrapper.get()
            poses = np.array(json_wrapper.get_keypoints_3d())
            poses = poses[:, joint_label_index, :] # y-axis flip..?
        else:
            continue

        # load audio
        wav_path = os.path.join(audio_path, '{}.wav'.format(name))
        if os.path.isfile(wav_path):
            # audio_raw, audio_sr = librosa.load(wav_path, mono=True, sr=16000, res_type='kaiser_fast')
            # mel audio feature
            audio_wrapper = AudioWrapper(wav_path)
            audio_feat, audio_raw = audio_wrapper.extract_audio_feat(len(poses), 0, len(poses))
        else:
            continue


        # process
        clips = [{'vid': name, 'clips': []},  # train
                 {'vid': name, 'clips': []}]  # validation

        # split
        if save_idx % 100 == 0:
            dataset_idx = 1  # validation
        else:
            dataset_idx = 0  # train

        # sentence preprocessing
        word_list = []
        for wi in range(len(subtitle)):
            word_s = float(subtitle[wi]['start_time'])
            word_e = float(subtitle[wi]['end_time'])
            word = subtitle[wi]['sentence_text'].strip()
            # word count with out space
            # word_count = len(word) - word.count(' ')
            word_tokens = word.split()
            total_char_count = 0
            for t_i, token in enumerate(word_tokens):
                # token = normalize_string(token)
                token = normalize_string_ko(token)
                total_char_count += len(token)
                word_tokens[t_i] = token
            
            prev_token_count = 0
            for t_i, token in enumerate(word_tokens):
                if len(token) > 0:
                    # split word with total char count in sentence and token_len ratio
                    new_s_time = word_s + (word_e - word_s) * (prev_token_count / total_char_count)
                    new_e_time = word_s + (word_e - word_s) * ((prev_token_count + len(token)) / total_char_count)
                    word_list.append([token, round(new_s_time, 4), round(new_e_time, 4)])
                    prev_token_count += len(token)

                    # previouse version (split word with equal time)
                    # new_s_time = word_s + (word_e - word_s) * t_i / len(word_tokens)
                    # new_e_time = word_s + (word_e - word_s) * (t_i + 1) / len(word_tokens)
                    # word_list.append([token, new_s_time, new_e_time])
        # print(word_list)

        # word preprocessing
        # word_list = []
        # for wi in range(len(subtitle)):
        #     word_s = float(subtitle[wi][0])
        #     word_e = float(subtitle[wi][1])
        #     word = subtitle[wi][2].strip()

        #     word_tokens = word.split()

        #     for t_i, token in enumerate(word_tokens):
        #         token = normalize_string(token)
        #         if len(token) > 0:
        #             new_s_time = word_s + (word_e - word_s) * t_i / len(word_tokens)
        #             new_e_time = word_s + (word_e - word_s) * (t_i + 1) / len(word_tokens)
        #             word_list.append([token, new_s_time, new_e_time])

        # save subtitles and skeletons
        poses = np.asarray(poses, dtype=np.float16)
        audio_raw = np.asarray(audio_raw, dtype=np.float16)
        audio_feat = np.asarray(audio_feat, dtype=np.float16)

        # print(f'poses.shape: {poses.shape}, audio_raw.shape: {audio_raw.shape}, len(word_list): {len(word_list)}')
        # print(f'audio_feat.shape: {audio_feat.shape}')
        # print(f'len keypoints_3d: {np.array(json_wrapper.get_keypoints_3d()).shape}')
        # print(f'audio duration: {len(audio_raw) / 16000}', json_wrapper.get_duration(), len(poses) / 30)
        # print(f'pose[0]: {poses.tolist()[0]}')
        # print(f'audio_raw[0]: {audio_raw.tolist()[0]}')
        # print(f'audio_feat[0]: {audio_feat.tolist()[0]}')
        clips[dataset_idx]['clips'].append(
            {'words': word_list,
            #  'poses': poses.tolist(),
             'skeletons_3d': poses,
             'audio_raw': audio_raw,
            'audio_feat': audio_feat,
            'start_frame_no': 0,
            'end_frame_no': len(poses) - 1,
            'start_time': 0,
            'end_time': json_wrapper.get_duration()
             })
        
        # poses_mirror = np.asarray(poses_mirror, dtype=np.float16)
        # clips[dataset_idx]['clips'].append(
        #     {'words': word_list,
        #      'poses': poses_mirror,
        #      'audio_raw': audio_raw
        #      })

        # write to db
        for i in range(2):
            with db[i].begin(write=True) as txn:
                if len(clips[i]['clips']) > 0:
                    k = '{:010}'.format(save_idx).encode('ascii')
                    v = pyarrow.serialize(clips[i]).to_buffer()
                    txn.put(k, v)

        all_poses.append(poses.reshape((-1, 30)))
        save_idx += 1

    # close db
    for i in range(2):
        db[i].sync()
        db[i].close()

    # calculate data mean
    all_poses = np.vstack(all_poses)
    pose_mean = np.mean(all_poses, axis=0, dtype=np.float64)
    pose_std = np.std(all_poses, axis=0, dtype=np.float64)

    print('data mean/std')
    print('data_mean:', str(["{:0.5f}".format(e) for e in pose_mean]).replace("'", ""))
    print('data_std:', str(["{:0.5f}".format(e) for e in pose_std]).replace("'", ""))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("db_path", type=Path)
    args = parser.parse_args()

    make_lmdb_gesture_dataset(args.db_path)

# data_mean: [-0.03138, 85.75205, 2.41944, 0.99564, -0.00111, 0.01194, 0.00169, 0.99865, 0.00775, -0.01174, -0.00776, 0.99481, 1.31403, 11.84144, -0.32539, 0.99551, 0.00822, 0.00489, -0.00944, 0.97781, 0.15469, -0.00321, -0.15624, 0.97941, -56.03576, 31.55597, -17.28595, 0.19382, -0.01879, -0.73267, 0.21954, 0.88659, -0.00434, 0.70373, -0.28352, 0.23952, 10.25930, -14.53623, 13.18367, 0.32494, -0.65823, 0.52997, 0.85662, 0.38615, -0.00255, -0.25492, 0.49138, 0.73453, -1.89513, -14.74314, 64.53655, 0.89218, 0.18793, -0.03377, -0.15467, 0.85145, 0.21401, 0.06266, -0.18852, 0.91192]
# data_std: [4.73436, 3.30040, 13.80028, 0.00843, 0.02392, 0.08907, 0.02268, 0.00219, 0.04630, 0.08942, 0.04569, 0.00867, 3.98747, 11.61510, 2.76606, 0.00585, 0.07557, 0.05601, 0.07859, 0.01983, 0.11537, 0.05160, 0.11539, 0.01892, 23.21881, 19.07393, 19.04333, 0.54988, 0.23940, 0.25612, 0.24814, 0.13507, 0.29312, 0.23463, 0.24013, 0.50428, 24.61702, 14.40351, 17.43484, 0.20707, 0.26829, 0.25577, 0.12168, 0.18785, 0.25882, 0.19479, 0.26249, 0.21721, 17.55170, 13.71203, 39.79642, 0.14564, 0.34459, 0.16612, 0.31180, 0.15579, 0.28952, 0.24026, 0.24732, 0.10023]