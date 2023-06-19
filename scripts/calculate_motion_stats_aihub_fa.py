import os
import lmdb
import numpy as np
import pyarrow
import math

from tqdm import tqdm
from data_loader.motion_preprocessor_aihub import MotionPreprocessor
from utils.data_utils import resample_pose_seq, calc_spectrogram_length_from_motion_length
from collections import defaultdict

def get_words_in_time_range(word_list, start_time, end_time):
    words = []

    for word in word_list:
        _, word_s, word_e = word[0], word[1], word[2]

        if word_s >= end_time:
            break

        if word_e <= start_time:
            continue

        words.append(word)

    return words

def calculate_data_mean(base_path):
    lmdb_path = os.path.join(base_path, 'lmdb_test')
    lmdb_env = lmdb.open(lmdb_path, readonly=True, lock=False)
    with lmdb_env.begin() as txn:
        n_videos = txn.stat()['entries']
    src_txn = lmdb_env.begin(write=False)
    cursor = src_txn.cursor()
    n_pose = 34
    subdivision_stride = 10
    skeleton_resampling_fps = 15
    pose_seq_list = []
    duration = []
    mean_pose = [-10.86856, 869.57773, 101.33229, -10.72149, 944.91927, 99.95321, -9.79247, 1122.75707, 61.54399, -10.58498, 1323.55203, 56.52225, -12.05896, 1460.32917, 87.05334, 25.78733, 1282.45543, 55.00527, 152.16606, 1289.76238, 58.01432, 228.05049, 1056.83990, 84.93741, 188.08313, 1001.43804, 233.00208, -46.68136, 1282.16211, 54.22307, -173.19778, 1287.35191, 55.57354, -249.45129, 1055.41187, 84.50893, -213.03483, 1003.46276, 232.51224, 80.05693, 869.41429, 102.42359, 110.07840, 465.02201, 97.89946, 122.33407, 110.19615, 44.14206, -101.79406, 869.74122, 100.24100, -120.06018, 464.83646, 87.92560, -122.39348, 109.82835, 34.72097, 137.66107, 54.95229, 180.11621, -143.90146, 53.92723, 169.55333]

    print(f"no. of videos: {n_videos}")
    no_clip = 0
    no_subdivision = 0
    no_sample = 0
    n_filtered_out = defaultdict(int)
    for key, value in tqdm(cursor):
        video = pyarrow.deserialize(value)
        vid = video['vid']
        clips = video['clips']
        for clip_idx, clip in enumerate(clips):
            no_clip += 1
            clip_s_t = clip['start_time']
            clip_e_t = clip['end_time']
            poses = clip['skeletons_3d']
            clip_audio = clip['audio_feat']
            clip_audio_raw = clip['audio_raw']
            clip_word_list = clip['words']

            pose_seq_list.append(poses)
            clip_skeleton = resample_pose_seq(poses, clip_e_t - clip_s_t, skeleton_resampling_fps)
            num_subdivision = math.floor((len(clip_skeleton) - n_pose) / subdivision_stride) + 1  # floor((K - (N+M)) / S) + 1
            duration.append(clip_e_t - clip_s_t)
            expected_audio_length = calc_spectrogram_length_from_motion_length(len(clip_skeleton), skeleton_resampling_fps)
            if not (abs(expected_audio_length - clip_audio.shape[1]) <= 5): # 'audio and skeleton lengths are different'
                num_subdivision = 0

            for i in range(num_subdivision):
                start_idx = i * subdivision_stride
                fin_idx = start_idx + n_pose

                sample_skeletons = clip_skeleton[start_idx:fin_idx]
                subdivision_start_time = clip_s_t + start_idx / skeleton_resampling_fps
                subdivision_end_time = clip_s_t + fin_idx / skeleton_resampling_fps
                sample_words = get_words_in_time_range(word_list=clip_word_list,
                                                            start_time=subdivision_start_time,
                                                            end_time=subdivision_end_time)
                # sample_words = [word for word in sample_words if word[0] != '<SIL>'] # --> 이거 넣어야하지만 못했음

                if len(sample_words) >= 2:
                    # filtering motion skeleton data
                    sample_skeletons, filtering_message = MotionPreprocessor(sample_skeletons, mean_pose).get()
                    is_correct_motion = (sample_skeletons != [])

                    if is_correct_motion:
                        no_sample += 1
                    else:
                        n_filtered_out[filtering_message] += 1
            no_subdivision += num_subdivision
    total_duration = sum(duration)
    avg_duration = total_duration / len(duration)

    # close db
    lmdb_env.close()
    print(f"no. of clips: {no_clip}")
    print(f"no. of subdivisions: {no_subdivision}")
    print(f"clip avg duration: {avg_duration} sec")
    print(f"clip total duration: {total_duration} sec")
    print(f"no. of samples: {no_sample}")
    print(f"n_filtered_out: {n_filtered_out}")

    # all_poses = np.vstack(pose_seq_list)
    # mean_pose = np.mean(all_poses, axis=0)

    # # mean dir vec
    # dir_vec = utils.data_utils_aihub.convert_pose_seq_to_dir_vec(all_poses)
    # mean_dir_vec = np.mean(dir_vec, axis=0)

    # # mean bone length
    # bone_lengths = []
    # for i, pair in enumerate(utils.data_utils_aihub.full_dir_vec_pairs):
    #     vec = all_poses[:, pair[1]] - all_poses[:, pair[0]]
    #     bone_lengths.append(np.mean(np.linalg.norm(vec, axis=1)))

    # print('mean pose', repr(mean_pose.flatten()))
    # print('mean directional vector', repr(mean_dir_vec.flatten()))
    # print('mean bone lengths', repr(bone_lengths))
    # print('total duration of the valid clips: {:.1f} h'.format(total_duration/3600))


if __name__ == '__main__':
    # import matplotlib
    # matplotlib.use('TkAgg')
    np.set_printoptions(precision=7, suppress=True)

    lmdb_base_path = './data/aihub/lmdb'
    calculate_data_mean(lmdb_base_path)
