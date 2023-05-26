import csv
import re

import librosa
import numpy as np
import torch
from scipy.interpolate import interp1d
from sklearn.preprocessing import normalize

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


skeleton_line_pairs = [(0, 1, 'b'), (1, 2, 'darkred'), (2, 3, 'r'), (3, 4, 'orange'), (1, 5, 'darkgreen'),
                       (5, 6, 'limegreen'), (6, 7, 'darkseagreen')]
# dir_vec_pairs = [(0, 1, 0.26), (1, 2, 0.18), (2, 3, 0.14), (1, 4, 0.22), (4, 5, 0.36),
#                  (5, 6, 0.33), (1, 7, 0.22), (7, 8, 0.36), (8, 9, 0.33)]  # adjacency and bone length
dir_vec_pairs = [(0, 1, 0.26), (1, 2, 0.18), (2, 3, 0.14), (2, 4, 0.22), (4, 5, 0.36),
                 (5, 6, 0.33), (2, 7, 0.22), (7, 8, 0.36), (8, 9, 0.33)]  # adjacency and bone length
full_dir_vec_pairs = [(0, 1, 0.26), (1, 2, 0.18), (2, 3, 0.14), (3, 4, 0.14),
                      (2, 5, 0.22), (5, 6, 0.36), (6, 7, 0.33), (7, 8, 0.33),
                        (2, 9, 0.22), (9, 10, 0.36), (10, 11, 0.33), (11, 12, 0.33),
                        (0, 13, 0.22), (13, 14, 0.36), (14, 15, 0.33),
                        (0, 16, 0.22), (16, 17, 0.36), (17, 18, 0.33),
                        (15, 19, 0.10), (18, 20, 0.10)]  # adjacency and bone length
full_skeleton_line_pairs = [(0, 1, 'b'), (1, 2, 'darkred'), (2, 3, 'r'), (3, 4, 'orange'),
                            (2, 5, 'darkgreen'), (5, 6, 'limegreen'), (6, 7, 'darkseagreen'),
                            (2, 9, 'darkgreen'), (9, 10, 'limegreen'), (10, 11, 'darkseagreen'),
                            (11, 12, 'darkseagreen'), (0, 13, 'darkgreen'), (13, 14, 'limegreen'),
                            (14, 15, 'darkseagreen'), (0, 16, 'darkgreen'), (16, 17, 'limegreen'),
                            (17, 18, 'darkseagreen'), (15, 19, 'darkgreen'), (18, 20, 'darkgreen')]

def normalize_string(s):
    """ lowercase, trim, and remove non-letter characters """
    s = s.lower().strip()
    s = re.sub(r"([,.!?])", r" \1 ", s)  # isolate some marks
    s = re.sub(r"(['])", r"", s)  # remove apostrophe
    s = re.sub(r"[^a-zA-Z,.!?]+", r" ", s)  # replace other characters with whitespace
    s = re.sub(r"\s+", r" ", s).strip()
    return s

class SubtitleWrapper:
    TIMESTAMP_PATTERN = re.compile('(\d+)?:?(\d{2}):(\d{2})[.,](\d{3})')

    def __init__(self, subtitle_path):
        self.subtitle = []
        self.load_tsv_subtitle(subtitle_path)

    def get(self):
        return self.subtitle

    def load_tsv_subtitle(self, subtitle_path):
        try:
            with open(subtitle_path) as file:
                tsv_file = csv.reader(file, delimiter="\t")
                for line in tsv_file:
                    self.subtitle.append(line)
        except FileNotFoundError:
            self.subtitle = None

    # convert timestamp to second
    def get_seconds(self, word_time_e):
        time_value = re.match(self.TIMESTAMP_PATTERN, word_time_e)
        if not time_value:
            print('wrong time stamp pattern')
            exit()

        values = list(map(lambda x: int(x) if x else 0, time_value.groups()))
        hours, minutes, seconds, milliseconds = values[0], values[1], values[2], values[3]

        return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000

def remove_tags_marks(text):
    reg_expr = re.compile('<.*?>|[.,:;!?]+')
    clean_text = re.sub(reg_expr, '', text)
    return clean_text


def extract_melspectrogram(y, sr=16000):
    melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=512, power=2)
    log_melspec = librosa.power_to_db(melspec, ref=np.max)  # mels x time
    log_melspec = log_melspec.astype('float16')
    return log_melspec


def calc_spectrogram_length_from_motion_length(n_frames, fps):
    ret = (n_frames / fps * 16000 - 1024) / 512 + 1
    return int(round(ret))


def resample_pose_seq(poses, duration_in_sec, fps):
    n = len(poses)
    x = np.arange(0, n)
    y = poses
    f = interp1d(x, y, axis=0, kind='linear', fill_value='extrapolate')
    expected_n = duration_in_sec * fps
    x_new = np.arange(0, n, n / expected_n)
    interpolated_y = f(x_new)
    if hasattr(poses, 'dtype'):
        interpolated_y = interpolated_y.astype(poses.dtype)
    return interpolated_y


def time_stretch_for_words(words, start_time, speech_speed_rate):
    for i in range(len(words)):
        if words[i][1] > start_time:
            words[i][1] = start_time + (words[i][1] - start_time) / speech_speed_rate
        words[i][2] = start_time + (words[i][2] - start_time) / speech_speed_rate

    return words


def make_audio_fixed_length(audio, expected_audio_length):
    n_padding = expected_audio_length - len(audio)
    if n_padding > 0:
        audio = np.pad(audio, (0, n_padding), mode='symmetric')
    else:
        audio = audio[0:expected_audio_length]
    return audio


def convert_dir_vec_to_pose(vec):
    vec = np.array(vec)

    if vec.shape[-1] != 3:
        vec = vec.reshape(vec.shape[:-1] + (-1, 3))

    if len(vec.shape) == 2:
        joint_pos = np.zeros((10, 3))
        for j, pair in enumerate(dir_vec_pairs):
            joint_pos[pair[1]] = joint_pos[pair[0]] + pair[2] * vec[j]
    elif len(vec.shape) == 3:
        joint_pos = np.zeros((vec.shape[0], 10, 3))
        for j, pair in enumerate(dir_vec_pairs):
            joint_pos[:, pair[1]] = joint_pos[:, pair[0]] + pair[2] * vec[:, j]
    elif len(vec.shape) == 4:  # (batch, seq, 9, 3)
        joint_pos = np.zeros((vec.shape[0], vec.shape[1], 10, 3))
        for j, pair in enumerate(dir_vec_pairs):
            joint_pos[:, :, pair[1]] = joint_pos[:, :, pair[0]] + pair[2] * vec[:, :, j]
    else:
        assert False

    return joint_pos


def convert_pose_seq_to_dir_vec(pose):
    print(f"convert_pose_seq_to_dir_vec: pose.shape {pose.shape}")
    if pose.shape[-1] != 3:
        pose = pose.reshape(pose.shape[:-1] + (-1, 3))

    if len(pose.shape) == 3:
        dir_vec = np.zeros((pose.shape[0], len(dir_vec_pairs), 3))
        for i, pair in enumerate(dir_vec_pairs):
            dir_vec[:, i] = pose[:, pair[1]] - pose[:, pair[0]]
            dir_vec[:, i, :] = normalize(dir_vec[:, i, :], axis=1)  # to unit length
    elif len(pose.shape) == 4:  # (batch, seq, ...)
        dir_vec = np.zeros((pose.shape[0], pose.shape[1], len(dir_vec_pairs), 3))
        for i, pair in enumerate(dir_vec_pairs):
            dir_vec[:, :, i] = pose[:, :, pair[1]] - pose[:, :, pair[0]]
        for j in range(dir_vec.shape[0]):  # batch
            for i in range(len(dir_vec_pairs)):
                dir_vec[j, :, i, :] = normalize(dir_vec[j, :, i, :], axis=1)  # to unit length
    else:
        assert False

    return dir_vec


def convert_dir_vec_to_pose_fullbody(vec):
    vec = np.array(vec)

    if vec.shape[-1] != 3:
        vec = vec.reshape(vec.shape[:-1] + (-1, 3))

    if len(vec.shape) == 2:
        joint_pos = np.zeros((21, 3))
        for j, pair in enumerate(full_dir_vec_pairs):
            joint_pos[pair[1]] = joint_pos[pair[0]] + pair[2] * vec[j]
    elif len(vec.shape) == 3:
        joint_pos = np.zeros((vec.shape[0], 21, 3))
        for j, pair in enumerate(full_dir_vec_pairs):
            joint_pos[:, pair[1]] = joint_pos[:, pair[0]] + pair[2] * vec[:, j]
    elif len(vec.shape) == 4:  # (batch, seq, 9, 3)
        joint_pos = np.zeros((vec.shape[0], vec.shape[1], 21, 3))
        for j, pair in enumerate(full_dir_vec_pairs):
            joint_pos[:, :, pair[1]] = joint_pos[:, :, pair[0]] + pair[2] * vec[:, :, j]
    else:
        assert False

    return joint_pos


def convert_pose_seq_to_dir_vec_fullbody(pose):
    if pose.shape[-1] != 3:
        pose = pose.reshape(pose.shape[:-1] + (-1, 3))

    if len(pose.shape) == 3:
        dir_vec = np.zeros((pose.shape[0], len(full_dir_vec_pairs), 3))
        for i, pair in enumerate(full_dir_vec_pairs):
            dir_vec[:, i] = pose[:, pair[1]] - pose[:, pair[0]]
            dir_vec[:, i, :] = normalize(dir_vec[:, i, :], axis=1)  # to unit length
    elif len(pose.shape) == 4:  # (batch, seq, ...)
        dir_vec = np.zeros((pose.shape[0], pose.shape[1], len(full_dir_vec_pairs), 3))
        for i, pair in enumerate(full_dir_vec_pairs):
            dir_vec[:, :, i] = pose[:, :, pair[1]] - pose[:, :, pair[0]]
        for j in range(dir_vec.shape[0]):  # batch
            for i in range(len(full_dir_vec_pairs)):
                dir_vec[j, :, i, :] = normalize(dir_vec[j, :, i, :], axis=1)  # to unit length
    else:
        assert False

    return dir_vec
