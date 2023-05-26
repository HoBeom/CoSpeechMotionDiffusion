import argparse
import glob
import json
import math
import os
import re
from pathlib import Path
import soundfile as sf

import joblib as jl
import librosa
import lmdb
import numpy as np
import pyarrow
from pymo.parsers import BVHParser
from pymo.preprocessing import *
from sklearn.pipeline import Pipeline
from utils.data_utils_aihub import (
    convert_dir_vec_to_pose,
    convert_pose_seq_to_dir_vec,
    resample_pose_seq,
    dir_vec_pairs,
    convert_pose_seq_to_dir_vec_fullbody,
    convert_dir_vec_to_pose_fullbody,
    full_dir_vec_pairs,
)
from sklearn.preprocessing import normalize

try:
    from rich.progress import track as tqdm
except ImportError:
    from tqdm import tqdm

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

def process_bvh(gesture_filename, tgt_fps=30, dump_pipeline=False):
    p = BVHParser()

    data_all = list()
    data_all.append(p.parse(gesture_filename))

    data_pipe = Pipeline(
        [
            ("dwnsampl", DownSampler(tgt_fps=tgt_fps, keep_all=False)),
            ('root', RootNormalizer()),
            ("jtsel", JointSelector(target_joints, include_root=False)),
            # ('mir', Mirror(axis='X', append=True)),
            ("np", Numpyfier()),
        ]
    )

    out_data = data_pipe.fit_transform(data_all)
    if dump_pipeline:
        jl.dump(data_pipe, os.path.join("./resource", "data_pipe.sav"))
    # euler -> rotation matrix
    out_data = out_data.reshape(
        (out_data.shape[0], out_data.shape[1], -1, 6)
    )  # 3 pos (XYZ), 3 rot (ZXY)
    out_matrix = np.zeros(
        (out_data.shape[0], out_data.shape[1], out_data.shape[2], 12)
    )  # 3 pos, 1 rot matrix (9 elements)
    for i in range(out_data.shape[0]):  # mirror
        for j in range(out_data.shape[1]):  # frames
            for k in range(out_data.shape[2]):  # joints
                out_matrix[i, j, k, :3] = out_data[i, j, k, :3]  # positions
                r = R.from_euler("ZXY", out_data[i, j, k, 3:], degrees=True)
                out_matrix[i, j, k, 3:] = r.as_matrix().flatten()  # rotations
    out_matrix = out_matrix.reshape((out_data.shape[0], out_data.shape[1], -1))

    return out_matrix[0]


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ]
    )


import matplotlib.pyplot as plt
from textwrap import wrap
import matplotlib.animation as animation
import datetime
import subprocess
import time


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

class AudioWrapper:
    def __init__(self, filepath):
        self.y, self.sr = librosa.load(
            filepath, mono=True, sr=16000, res_type="kaiser_fast"
        )
        self.n = len(self.y)

    def extract_audio_feat(
        self, video_total_frames, video_start_frame, video_end_frame
    ):
        # roi
        start_frame = math.floor(video_start_frame / video_total_frames * self.n)
        end_frame = math.ceil(video_end_frame / video_total_frames * self.n)
        y_roi = self.y[start_frame:end_frame]

        # feature extraction
        melspec = librosa.feature.melspectrogram(
            y=y_roi, sr=self.sr, n_fft=1024, hop_length=512, power=2
        )
        log_melspec = librosa.power_to_db(melspec, ref=np.max)  # mels x time

        log_melspec = log_melspec.astype("float16")
        y_roi = y_roi.astype("float16")

        # DEBUG
        # print('spectrogram shape: ', log_melspec.shape)

        return log_melspec, y_roi


def create_video_and_save(
    save_path,
    iter_idx,
    prefix,
    target,
    output,
    mean_data,
    title,
    audio=None,
    aux_str=None,
    clipping_to_shortest_stream=False,
    delete_audio_file=True,
):
    print("rendering a video...")
    start = time.time()

    fig = plt.figure(figsize=(8, 4))
    axes = [
        fig.add_subplot(1, 2, 1, projection="3d"),
        fig.add_subplot(1, 2, 2, projection="3d"),
    ]
    axes[0].view_init(elev=20, azim=-60)
    axes[1].view_init(elev=20, azim=-60)
    fig_title = title

    if aux_str:
        fig_title += "\n" + aux_str
    fig.suptitle("\n".join(wrap(fig_title, 75)), fontsize="medium")

    # un-normalization and convert to poses
    mean_data = mean_data.flatten()
    output = output + mean_data
    # output_poses = convert_dir_vec_to_pose(output)
    output_poses = convert_dir_vec_to_pose_fullbody(output)
    target_poses = None
    if target is not None:
        target = target + mean_data
        # target_poses = convert_dir_vec_to_pose(target)
        target_poses = convert_dir_vec_to_pose_fullbody(target)

    def animate(i):
        for k, name in enumerate(["human", "generated"]):
            if name == "human" and target is not None and i < len(target):
                pose = target_poses[i]
            elif name == "generated" and i < len(output):
                pose = output_poses[i]
            else:
                pose = None

            if pose is not None:
                axes[k].clear()
                for j, pair in enumerate(full_dir_vec_pairs):
                    axes[k].plot(
                        [pose[pair[0], 0], pose[pair[1], 0]],
                        [pose[pair[0], 2], pose[pair[1], 2]],
                        [pose[pair[0], 1], pose[pair[1], 1]],
                        zdir="z",
                        linewidth=5,
                    )
                # axes[k].set_xlim3d(-0.5, 0.5)
                # axes[k].set_ylim3d(0.5, -0.5)
                # axes[k].set_zlim3d(0.0, 1.0)
                axes[k].set_xlim3d(-1.5, 1.5)
                axes[k].set_ylim3d(1.5, -1.5)
                axes[k].set_zlim3d(-1.5, 1.5)
                axes[k].set_xlabel("x")
                axes[k].set_ylabel("z")
                axes[k].set_zlabel("y")
                axes[k].set_title("{} ({}/{})".format(name, i + 1, len(output)))
                # axes[k].axis('off')

    if target is not None:
        num_frames = max(len(target), len(output))
    else:
        num_frames = len(output)
    ani = animation.FuncAnimation(
        fig, animate, interval=30, frames=num_frames, repeat=False
    )

    # show audio
    audio_path = None
    if audio is not None:
        assert len(audio.shape) == 1  # 1-channel, raw signal
        audio = audio.astype(np.float32)
        sr = 16000
        audio_path = "{}/{}.wav".format(save_path, iter_idx)
        sf.write(audio_path, audio, sr)

    # save video
    try:
        video_path = "{}/temp_{}.mp4".format(save_path, iter_idx)
        ani.save(
            video_path, fps=15, dpi=150, codec="mpeg4"
        )  # dpi 150 for a higher resolution
        del ani
        plt.close(fig)
    except RuntimeError:
        assert False, "RuntimeError"

    # merge audio and video
    if audio is not None:
        merged_video_path = "{}/{}_{}.mp4".format(save_path, prefix, iter_idx)
        cmd = [
            "ffmpeg",
            "-loglevel",
            "panic",
            "-y",
            "-i",
            video_path,
            "-i",
            audio_path,
            "-strict",
            "-2",
            merged_video_path,
        ]
        if clipping_to_shortest_stream:
            cmd.insert(len(cmd) - 1, "-shortest")
        subprocess.call(cmd)
        if delete_audio_file:
            os.remove(audio_path)
        os.remove(video_path)

    print("done, took {:.1f} seconds".format(time.time() - start))
    return output_poses, target_poses


def make_lmdb_gesture_dataset(base_path, save_video=False):
    gesture_path = os.path.join(base_path, "bvh")
    text_path = os.path.join(base_path, 'json')
    audio_path = os.path.join(base_path, "wav")

    all_poses = []
    all_dir_vecs = []
    error_bvh_files = []
    bvh_files = sorted(glob.glob(os.path.join(gesture_path, "*.bvh")))
    print("bvh_files:", len(bvh_files))
    if len(bvh_files) == 0:
        bvh_files = sorted(glob.glob(os.path.join(base_path, "*.bvh")))
        if len(bvh_files) == 0:
            print(f"no bvh files found in {base_path}")
            return
    for bvh_file in tqdm(bvh_files):
        name = os.path.split(bvh_file)[1][:-4]
        print("file name:", name)
        # poses process_bvh(bvh_file)
        # # save subtitles and skeletons
        # poses = np.asarray(poses, dtype=np.float16)
        # # test reshape and recover
        # try:
        #     poses = poses.reshape((-1, 10, 6))
        # except:
        #     print("poses.shape:", poses.shape)
        #     error_bvh_files.append(bvh_file)
        #     continue

        # dir_vec = convert_pose_seq_to_dir_vec(poses[:, :, :3])


        json_path = os.path.join(text_path, name + '.json')
        if os.path.isfile(json_path):
            json_wrapper = JsonSubtitleWrapper(json_path)
            # subtitle = json_wrapper.get()
            kpt_3d = np.array(json_wrapper.get_keypoints_3d())
            print("kpt_3d.shape:", kpt_3d.shape)
            # kpt_3d = kpt_3d[:, joint_label_index, :]
        else:
            continue

        # dir_vec = convert_pose_seq_to_dir_vec(kpt_3d)
        dir_vec = convert_pose_seq_to_dir_vec_fullbody(kpt_3d)
        print("dir_vec.shape:", dir_vec.shape)
        dir_vec = dir_vec.reshape((-1, 20*3))
        all_dir_vecs.append(dir_vec)
        kpt_3d = kpt_3d.reshape((-1, 21*3))
        all_poses.append(kpt_3d)
        print("kpt_3d.shape:", kpt_3d.shape)
        print("dir_vec.shape:", dir_vec.shape)

        if save_video:
            wav_path = os.path.join(audio_path, "{}.wav".format(name))
            audio_wrapper = AudioWrapper(wav_path)
            audio_feat, audio_raw = audio_wrapper.extract_audio_feat(
                len(kpt_3d), 0, len(kpt_3d)
            )
            aux_str = "({}, time: {}-{})".format(name, str(0), str(len(kpt_3d) // 30))
            # mean_dir_vec = [-0.01716, 0.98585, -0.15346, 0.00465, 0.99418, 
            #                 -0.07478, -0.00864, 0.94903, 0.25143, -0.68102, 
            #                 0.71283, -0.07691, -0.33603, -0.87266, 0.17670, 
            #                 0.15803, -0.22567, 0.59774, 0.68220, 0.70908, 
            #                 -0.07005, 0.35725, -0.86622, 0.13643, -0.15108, 
            #                 -0.16974, 0.62026]
            mean_dir_vec = [-0.01716, 0.98585, -0.15346, 0.00465, 0.99418, -0.07478, -0.00864, 0.94903, 0.25143, -0.68102, 0.71283, -0.07691, -0.33603, -0.87266, 0.17670, 0.15803, -0.22567, 0.59774, 0.68220, 0.70908, -0.07005, 0.35725, -0.86622, 0.13643, -0.15108, -0.16974, 0.62026]
            mean_dir_vec = [-0.00111, 0.99864, -0.00776, -0.02409, 0.97080, -0.21897, 0.00465, 0.99418, -0.07478, -0.00864, 0.94903, 0.25143, 0.23448, 0.96345, -0.09320, 0.98098, 0.01497, -0.00621, 0.35725, -0.86622, 0.13643, -0.15108, -0.16974, 0.62026, -0.22573, 0.96575, -0.08658, -0.98050, 0.00949, -0.02888, -0.33603, -0.87266, 0.17670, 0.15803, -0.22567, 0.59774, 0.99563, 0.00169, -0.01174, 0.09400, -0.99062, -0.02207, 0.04146, -0.98580, -0.14363, -0.99563, -0.00169, 0.01174, -0.05012, -0.99466, -0.05283, -0.00148, -0.98434, -0.14879, 0.12960, -0.37522, 0.91059, -0.14808, -0.38080, 0.89982]
            create_video_and_save(
                ".",
                1,
                "long",
                dir_vec,
                dir_vec,
                np.array(mean_dir_vec).reshape((-1, 3)),
                "",
                audio=audio_raw,
                aux_str=aux_str,
            )
            return 
    # calculate data mean
    all_poses = np.vstack(all_poses)
    pose_mean = np.mean(all_poses, axis=0, dtype=np.float64)
    pose_std = np.std(all_poses, axis=0, dtype=np.float64)

    all_dir_vecs = np.vstack(all_dir_vecs)
    dir_vec_mean = np.mean(all_dir_vecs, axis=0, dtype=np.float64)
    dir_vec_std = np.std(all_dir_vecs, axis=0, dtype=np.float64)

    print("data mean/std")
    print("data_mean:", str(["{:0.5f}".format(e) for e in pose_mean]).replace("'", ""))
    print("data_std:", str(["{:0.5f}".format(e) for e in pose_std]).replace("'", ""))

    print(
        "dir_vec_mean:",
        str(["{:0.5f}".format(e) for e in dir_vec_mean]).replace("'", ""),
    )
    print(
        "dir_vec_std:", str(["{:0.5f}".format(e) for e in dir_vec_std]).replace("'", "")
    )

    print("error_bvh_files:", error_bvh_files)
    with open(os.path.join(base_path, "error_bvh_files.txt"), "w") as f:
        f.write("\n".join(error_bvh_files))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("db_path", type=Path)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    make_lmdb_gesture_dataset(args.db_path, args.save)

# from bvh
# data_mean: [-0.03138, 85.75205, 2.41944, 0.99564, -0.00111, 0.01194, 0.00169, 0.99865, 0.00775, -0.01174, -0.00776, 0.99481, 1.31403, 11.84144, -0.32539, 0.99551, 0.00822, 0.00489, -0.00944, 0.97781, 0.15469, -0.00321, -0.15624, 0.97941, -56.03576, 31.55597, -17.28595, 0.19382, -0.01879, -0.73267, 0.21954, 0.88659, -0.00434, 0.70373, -0.28352, 0.23952, 10.25930, -14.53623, 13.18367, 0.32494, -0.65823, 0.52997, 0.85662, 0.38615, -0.00255, -0.25492, 0.49138, 0.73453, -1.89513, -14.74314, 64.53655, 0.89218, 0.18793, -0.03377, -0.15467, 0.85145, 0.21401, 0.06266, -0.18852, 0.91192]
# data_std: [4.73436, 3.30040, 13.80028, 0.00843, 0.02392, 0.08907, 0.02268, 0.00219, 0.04630, 0.08942, 0.04569, 0.00867, 3.98747, 11.61510, 2.76606, 0.00585, 0.07557, 0.05601, 0.07859, 0.01983, 0.11537, 0.05160, 0.11539, 0.01892, 23.21881, 19.07393, 19.04333, 0.54988, 0.23940, 0.25612, 0.24814, 0.13507, 0.29312, 0.23463, 0.24013, 0.50428, 24.61702, 14.40351, 17.43484, 0.20707, 0.26829, 0.25577, 0.12168, 0.18785, 0.25882, 0.19479, 0.26249, 0.21721, 17.55170, 13.71203, 39.79642, 0.14564, 0.34459, 0.16612, 0.31180, 0.15579, 0.28952, 0.24026, 0.24732, 0.10023]

# kpt 3d half
# data_mean: [-0.31384, 857.52072, 24.19362, -4.49959, 1097.70250, -12.71143, -3.57232, 1288.79640, -27.60548, -4.90305, 1420.79526, 7.83838, -153.03711, 1252.81706, -30.05787, -240.26356, 1031.60934, 14.64896, -207.09588, 973.17363, 140.88095, 145.48783, 1253.36046, -27.99284, 237.48327, 1033.35510, 5.17584, 205.38234, 985.67803, 136.09054]
# data_std: [47.34222, 33.00330, 138.00333, 47.47163, 47.80961, 139.23748, 50.17617, 73.16897, 140.07845, 53.33127, 77.61282, 137.83575, 57.85619, 71.08291, 144.08834, 67.41710, 84.42622, 136.80725, 102.80658, 161.77922, 159.84972, 58.80085, 74.16703, 148.04524, 60.97007, 88.16441, 153.35891, 104.44696, 163.98093, 167.11714]
# dir_vec_mean: [-0.01716, 0.98585, -0.15346, 0.00465, 0.99418, -0.07478, -0.00864, 0.94903, 0.25143, -0.68102, 0.71283, -0.07691, -0.33603, -0.87266, 0.17670, 0.15803, -0.22567, 0.59774, 0.68220, 0.70908, -0.07005, 0.35725, -0.86622, 0.13643, -0.15108, -0.16974, 0.62026]
# dir_vec_std: [0.02575, 0.00993, 0.05906, 0.03902, 0.00562, 0.06657, 0.07435, 0.05457, 0.16599, 0.08063, 0.07403, 0.10089, 0.18727, 0.14815, 0.19316, 0.37546, 0.59583, 0.26614, 0.08990, 0.08275, 0.10938, 0.20502, 0.16213, 0.18734, 0.35065, 0.62341, 0.22812]

# edit vec_pairs
# data_mean: [-0.31384, 857.52072, 24.19362, -4.49959, 1097.70250, -12.71143, -3.57232, 1288.79640, -27.60548, -4.90305, 1420.79526, 7.83838, -153.03711, 1252.81706, -30.05787, -240.26356, 1031.60934, 14.64896, -207.09588, 973.17363, 140.88095, 145.48783, 1253.36046, -27.99284, 237.48327, 1033.35510, 5.17584, 205.38234, 985.67803, 136.09054]
# data_std: [47.34222, 33.00330, 138.00333, 47.47163, 47.80961, 139.23748, 50.17617, 73.16897, 140.07845, 53.33127, 77.61282, 137.83575, 57.85619, 71.08291, 144.08834, 67.41710, 84.42622, 136.80725, 102.80658, 161.77922, 159.84972, 58.80085, 74.16703, 148.04524, 60.97007, 88.16441, 153.35891, 104.44696, 163.98093, 167.11714]
# dir_vec_mean: [-0.01716, 0.98585, -0.15346, 0.00465, 0.99418, -0.07478, -0.00864, 0.94903, 0.25143, -0.68102, 0.71283, -0.07691, -0.33603, -0.87266, 0.17670, 0.15803, -0.22567, 0.59774, 0.68220, 0.70908, -0.07005, 0.35725, -0.86622, 0.13643, -0.15108, -0.16974, 0.62026]
# dir_vec_std: [0.02575, 0.00993, 0.05906, 0.03902, 0.00562, 0.06657, 0.07435, 0.05457, 0.16599, 0.08063, 0.07403, 0.10089, 0.18727, 0.14815, 0.19316, 0.37546, 0.59583, 0.26614, 0.08990, 0.08275, 0.10938, 0.20502, 0.16213, 0.18734, 0.35065, 0.62341, 0.22812]

# full vec_pairs
# data_mean: [-0.31384, 857.52072, 24.19362, -0.38050, 931.80871, 23.62638, -4.49959, 1097.70250, -12.71143, -3.57232, 1288.79640, -27.60548, -4.90305, 1420.79526, 7.83838, 31.75970, 1251.13046, -27.70294, 145.48783, 1253.36046, -27.99284, 237.48327, 1033.35510, 5.17584, 205.38234, 985.67803, 136.09054, -39.39186, 1251.44162, -26.94874, -153.03711, 1252.81706, -30.05787, -240.26356, 1031.60934, 14.64896, -207.09588, 973.17363, 140.88095, 88.82028, 857.65733, 23.18511, 126.89653, 455.62226, 14.35012, 141.53243, 110.67839, -35.92447, -89.44792, 857.38405, 25.20212, -109.68078, 453.71262, 3.78048, -110.38907, 109.28719, -48.44116, 160.97959, 55.84048, 97.29841, -132.66095, 53.62018, 83.19123]
# data_std: [47.34222, 33.00330, 138.00333, 47.39458, 35.61980, 137.73704, 47.47163, 47.80961, 139.23748, 50.17617, 73.16897, 140.07845, 53.33127, 77.61282, 137.83575, 49.75796, 70.16263, 140.66259, 58.80085, 74.16703, 148.04524, 60.97007, 88.16441, 153.35891, 104.44696, 163.98093, 167.11714, 49.64497, 69.60028, 139.48996, 57.85619, 71.08291, 144.08834, 67.41710, 84.42622, 136.80725, 102.80658, 161.77922, 159.84972, 47.81690, 32.73232, 139.94995, 41.67416, 25.28186, 126.54025, 42.66873, 15.72571, 125.46550, 47.25517, 33.39430, 136.49598, 44.08132, 25.35851, 130.99747, 38.22208, 14.11571, 132.81705, 48.76196, 13.42010, 127.34661, 40.94056, 12.31055, 135.04605]
# dir_vec_mean: [-0.00111, 0.99864, -0.00776, -0.02409, 0.97080, -0.21897, 0.00465, 0.99418, -0.07478, -0.00864, 0.94903, 0.25143, 0.23448, 0.96345, -0.09320, 0.98098, 0.01497, -0.00621, 0.35725, -0.86622, 0.13643, -0.15108, -0.16974, 0.62026, -0.22573, 0.96575, -0.08658, -0.98050, 0.00949, -0.02888, -0.33603, -0.87266, 0.17670, 0.15803, -0.22567, 0.59774, 0.99563, 0.00169, -0.01174, 0.09400, -0.99062, -0.02207, 0.04146, -0.98580, -0.14363, -0.99563, -0.00169, 0.01174, -0.05012, -0.99466, -0.05283, -0.00148, -0.98434, -0.14879, 0.12960, -0.37522, 0.91059, -0.14808, -0.38080, 0.89982]
# dir_vec_std: [0.02392, 0.00218, 0.04569, 0.03062, 0.02261, 0.08696, 0.03902, 0.00562, 0.06657, 0.07435, 0.05457, 0.16599, 0.04966, 0.01397, 0.07375, 0.02367, 0.10704, 0.15935, 0.20502, 0.16213, 0.18734, 0.35065, 0.62341, 0.22812, 0.05449, 0.01449, 0.07544, 0.01780, 0.13509, 0.13830, 0.18727, 0.14815, 0.19316, 0.37546, 0.59583, 0.26614, 0.00842, 0.02268, 0.08942, 0.06501, 0.01039, 0.07077, 0.05366, 0.01590, 0.05221, 0.00842, 0.02268, 0.08942, 0.05515, 0.00691, 0.04753, 0.05698, 0.01668, 0.07354, 0.10195, 0.04596, 0.02709, 0.14445, 0.03746, 0.03350]